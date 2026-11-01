const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const cuda = @import("cuda");

const ptx: [*c]const u8 = @embedFile("kernels.ptx");

const GPU = struct {
    device: cuda.CUdevice = 0,
    context: cuda.CUcontext = null,
    library: cuda.CUlibrary = null,
    // Kernels
    kernels: std.StringHashMap(cuda.CUfunction),

    pub fn init(alloc: std.mem.Allocator) !GPU {
        var gpu: GPU = .{ .kernels = .init(alloc) };
        try checkdrv(cuda.cuInit(0));
        try checkdrv(cuda.cuDeviceGet(&gpu.device, 0));
        try checkdrv(cuda.cuCtxCreate(&gpu.context, null, 0, gpu.device));
        try checkdrv(cuda.cuLibraryLoadData(&gpu.library, ptx, null, null, 0, null, null, 0));

        // Load functions
        var count: c_uint = 0;
        try checkdrv(cuda.cuLibraryGetKernelCount(&count, gpu.library));
        const kernels = try alloc.alloc(cuda.CUkernel, @intCast(count)); // @leak
        try checkdrv(cuda.cuLibraryEnumerateKernels(@ptrCast(kernels.ptr), count, gpu.library));
        for (kernels) |kernel| {
            var name: [*c]const u8 = null;
            try checkdrv(cuda.cuKernelGetName(&name, kernel));
            try gpu.kernels.put(std.mem.span(name), @ptrCast(kernel));
        }

        return gpu;
    }
};

fn DeviceSlice(comptime T: type) type {
    return struct {
        ptr: [*]T,
        len: usize,

        pub fn sub(self: @This(), start: usize, end: usize) @This() {
            assert(start < end and end <= self.len);
            return .{ .ptr = self.ptr + start, .len = end - start };
        }

        pub fn byteSize(self: @This()) usize {
            return self.len * @sizeOf(T);
        }

        pub fn copyFromHost(self: @This(), host_slice: []T) !void {
            assert(host_slice.len == self.len);
            try check(cuda.cudaMemcpy(self.ptr, host_slice.ptr, self.byteSize(), cuda.cudaMemcpyHostToDevice));
        }

        pub fn copyToHost(self: @This(), host_slice: []T) !void {
            assert(host_slice.len == self.len);
            try check(cuda.cudaMemcpy(host_slice.ptr, self.ptr, self.byteSize(), cuda.cudaMemcpyDeviceToHost));
        }
    };
}


pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    const gpu: GPU = try .init(arena);
    try unified_memory_example(io, gpu, 1024);
    try explicit_mem_example(io, gpu, 1024);
}

pub fn unified_memory_example(io: std.Io, gpu: GPU, vectorLength: c_uint) !void {
    var a: []f32 = undefined;
    var b: []f32 = undefined;
    var c: []f32 = undefined;
    a.len = @intCast(vectorLength);
    b.len = @intCast(vectorLength);
    c.len = @intCast(vectorLength);
    try check(cuda.cudaMallocManaged(@ptrCast(&a.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cudaFree(a.ptr);
    try check(cuda.cudaMallocManaged(@ptrCast(&b.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cudaFree(b.ptr);
    try check(cuda.cudaMallocManaged(@ptrCast(&c.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cudaFree(c.ptr);

    init_array(io, a);
    init_array(io, b);

    // Launch
    var params = [_]?*anyopaque{@ptrCast(&a.ptr), @ptrCast(&b.ptr), @ptrCast(&c.ptr), @constCast(&vectorLength)};
    const vec_add = gpu.kernels.get("vec_add").?;
    const threads = 256;
    var config: cuda.cudaLaunchConfig_t = .{
        .gridDim = dim3(@divCeil(vectorLength, threads), 1, 1),
        .blockDim = dim3(threads, 1, 1)
    };
    try check(cuda.cudaLaunchKernelExC(&config, vec_add, @ptrCast(&params)));
    try check(cuda.cudaDeviceSynchronize());

    assert(c.len > 0);
    for (0..c.len) |i| {
        std.debug.print("{d} + {d} = {d}\n", .{a[i], b[i], c[i]});
    }
}

pub fn explicit_mem_example(io: std.Io, gpu: GPU, vectorLength: c_uint) !void {
    var cudaHostAllocator: CudaHostAllocator = .{};
    var hostArena = std.heap.ArenaAllocator.init(cudaHostAllocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    const a = try hostAlloc.alloc(f32, vectorLength);
    const b = try hostAlloc.alloc(f32, vectorLength);
    const c = try hostAlloc.alloc(f32, vectorLength);

    init_array(io, a);
    init_array(io, b);

    // One allocation for all the vectors
    var deviceBuff: [*]f32 = undefined;
    try check(cuda.cudaMalloc(@ptrCast(&deviceBuff), vectorLength * @sizeOf(f32) * 3));
    defer _ = cuda.cudaFree(deviceBuff);

    var devA: DeviceSlice(f32) = .{ .ptr = deviceBuff, .len = vectorLength };
    var devB: DeviceSlice(f32) = .{ .ptr = deviceBuff + vectorLength, .len = vectorLength };
    var devC: DeviceSlice(f32) = .{ .ptr = deviceBuff + vectorLength * 2, .len = vectorLength };

    try devA.copyFromHost(a);
    try devB.copyFromHost(b);
    try check(cuda.cudaMemset(devC.ptr, 0, devC.byteSize()));

    // Launch
    var params = [_]?*anyopaque{@ptrCast(&devA.ptr), @ptrCast(&devB.ptr), @ptrCast(&devC.ptr), @constCast(&vectorLength)};
    const vec_add = gpu.kernels.get("vec_add").?;
    const threads = 256;
    var config: cuda.cudaLaunchConfig_t = .{
        .gridDim = dim3(@divCeil(vectorLength, threads), 1, 1),
        .blockDim = dim3(threads, 1, 1)
    };
    try check(cuda.cudaLaunchKernelExC(&config, vec_add, @ptrCast(&params)));
    try check(cuda.cudaDeviceSynchronize());

    try devC.copyToHost(c);

    assert(c.len > 0);
    for (0..c.len) |i| {
        std.debug.print("{d} + {d} = {d}\n", .{a[i], b[i], c[i]});
    }
}

fn init_array(io: std.Io,  a: []f32) void {
    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    for (a) |*x| {
        x.* = random.float(f32);
    }
}

fn check(r: cuda.cudaError_t) !void {
    if (r != cuda.cudaSuccess) {
        std.debug.print("CUDA runtime error: {s}\n", .{cuda.cudaGetErrorName(r)});
        return error.CudaError;
    }
}

fn checkdrv(r: cuda.CUresult) !void {
    if (r != cuda.CUDA_SUCCESS) {
        var name: [*c]const u8 = null;
        _ = cuda.cuGetErrorName(r, &name);
        if (name == null) name = "error not found";
        std.debug.print("CUDA error: {s}\n", .{name});
        return error.CudaError;
    }
}

pub fn dim3(x: c_uint, y: c_uint, z: c_uint) cuda.dim3 {
    return  .{ .x = x, .y = y, .z = z};
}

// Nvidia recommends allocating memory using cudaMallocHost API when 
// buffers will be used for copies between CPU and GPU memory.
// This is because it allocates page-locked memory (it will never be swapped to disk 
// or be physically rellocated by the operating system)
// This struct implements the zig allocator interface which allows it to interact with other
// zig allocators (e.g the ArenaAllocator)
pub const CudaHostAllocator = struct {
    pub fn allocator(self: *CudaHostAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .free = free,
                // not supported
                .resize = resize, .remap = remap,
            },
        };
    }

    pub fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = alignment;
        _ = ret_addr;

        var res: ?*anyopaque = undefined;
        const ok = cuda.cudaMallocHost(&res, @intCast(len));
        if (ok != cuda.CUDA_SUCCESS) {
            return null;
        }
        return @ptrCast(res);
    }

    pub fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        _ = ctx; _ = alignment; _ = ret_addr; 
        _ = cuda.cudaFreeHost(@ptrCast(memory.ptr));
    }

    pub fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx; _ = memory; _ = alignment; _ = new_len; _ = ret_addr;
        // resize not supported
        return false;
    }

    pub fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx; _ = memory; _ = alignment; _ = new_len; _ = ret_addr;
        // remap not supported
        return null;
    }
};
