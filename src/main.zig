const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const cuda = @import("cuda");

const ptx: [*c]const u8 = @embedFile("kernels.ptx");

const GPU = struct {
    device: cuda.CUdevice = 0,
    context: cuda.CUcontext = null,
    library: cuda.CUlibrary = null,
    // Host allocator that uses pinned memory
    host_allocator: CudaHostAllocator = .{},
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

    pub fn malloc(_: GPU, comptime T: type, size: usize) !DeviceSlice(T) {
        var ptr: [*]T = undefined;
        try check(cuda.cudaMalloc(@ptrCast(&ptr), size * @sizeOf(T)));
        return .init(ptr, size);
    }

    pub fn mallocManaged(_: GPU, comptime T: type, size: usize) ![]T {
        var ptr: [*]T = undefined;
        try check(cuda.cudaMallocManaged(@ptrCast(&ptr), size*@sizeOf(f32), cuda.cudaMemAttachGlobal));
        return ptr[0..size];
    }

    pub fn free(_: GPU, ptr: *anyopaque) void {
        _ = cuda.cudaFree(ptr);
    }
};

fn DeviceSlice(comptime T: type) type {
    return struct {
        ptr: [*]T,
        len: usize,

        pub fn init(ptr: [*]T, len: usize) @This() {
            return .{ .ptr = ptr, .len = len };
        }

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


pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var gpu: GPU = try .init(arena);
    try unified_memory_example(io, gpu, 1024);
    try explicit_mem_example(io, &gpu, 1024);
    try saxpy_grid_stride_problem(io, &gpu);
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

pub fn unified_memory_example(io: std.Io, gpu: GPU, vectorLength: c_uint) !void {
    var a = try gpu.mallocManaged(f32, vectorLength);
    defer gpu.free(a.ptr);
    var b = try gpu.mallocManaged(f32, vectorLength);
    defer gpu.free(b.ptr);
    var c = try gpu.mallocManaged(f32, vectorLength);
    defer gpu.free(c.ptr);

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    init_array(random, a);
    init_array(random, b);

    // Launch
    var params = [_]?*anyopaque{@ptrCast(&a.ptr), @ptrCast(&b.ptr), @ptrCast(&c.ptr), @constCast(&vectorLength)};
    const vec_add = gpu.kernels.get("vec_add").?;
    const threads = 256;
    var config: cuda.cudaLaunchConfig_t = .{
        .gridDim = dim3(@divCeil(vectorLength, threads), 1, 1),
        .blockDim = dim3(threads, 1, 1)
    };
    try check(cuda.cudaLaunchKernelExC(&config, vec_add, &params));
    try check(cuda.cudaDeviceSynchronize());

    assert(c.len > 0);
    for (0..c.len) |i| {
        if (@abs(a[i] + b[i] - c[i]) > 0.00001) {
            std.debug.print("Index {} mismatch: {} != {}\n", .{i, a[i]+b[i], c[i]});
            return error.ValuesDontMatch;
        }
    }
    std.debug.print("Results match!\n", .{});
}

pub fn explicit_mem_example(io: std.Io, gpu: *GPU, vectorLength: c_uint) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    const a = try hostAlloc.alloc(f32, vectorLength);
    const b = try hostAlloc.alloc(f32, vectorLength);
    const c = try hostAlloc.alloc(f32, vectorLength);

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    init_array(random, a);
    init_array(random, b);

    // One allocation for all the vectors
    const deviceBuff = try gpu.malloc(f32, vectorLength*3);
    defer gpu.free(deviceBuff.ptr);

    var devA = deviceBuff.sub(0, vectorLength);
    var devB = deviceBuff.sub(vectorLength, vectorLength*2);
    var devC = deviceBuff.sub(vectorLength*2, deviceBuff.len);

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
    try check(cuda.cudaLaunchKernelExC(&config, vec_add, &params));
    try check(cuda.cudaDeviceSynchronize());

    try devC.copyToHost(c);

    assert(c.len > 0);
    for (0..c.len) |i| {
        if (@abs(a[i] + b[i] - c[i]) > 0.00001) {
            std.debug.print("Index {} mismatch: {} != {}\n", .{i, a[i]+b[i], c[i]});
            return error.ValuesDontMatch;
        }
    }
    std.debug.print("Results match!\n", .{});
}

fn init_array(random: std.Random,  a: []f32) void {
    for (a) |*x| {
        x.* = random.float(f32);
    }
}

// Problem statement:
// 01 — SAXPY with a grid-stride loop ★☆☆☆☆
// Compute y[i] = a * x[i] + y[i] for n = 2^24 + 37 floats.
// Constraint: launch a fixed grid of 4 × (number of SMs) blocks of 256 threads. That's far fewer threads than elements, so each thread has to process several of them with a grid-stride loop.
// Measure: effective bandwidth (3 × n × 4 bytes / time) and compare it with your GPU's peak.
// Stretch: sweep 1–32 blocks per SM and find where bandwidth saturates. Then add a float4 version that handles the tail correctly.
pub fn saxpy_grid_stride_problem(io: std.Io, gpu: *GPU) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var n: u32 = std.math.pow(u32, 2, 24) + 37;
    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();

    const x = try hostAlloc.alloc(f32, n);
    const y = try hostAlloc.alloc(f32, n);
    var a = 0.1 + (100.0 - 0.1) * random.float(f32); // random float between 0.1 and 100

    init_array(random, x);
    init_array(random, y);

    // Since we are using float4 in one of the kernels we need to align the start of y to a multiple of 16 bytes.
    // So we find the next multiple of 4 >= n because 4*@sizeOf(f32) = 16.
    // devX is aligned because cudaMalloc guarantees a minimum alignment of 256 bytes (which is a multiple of 16)
    //
    // Another way to do it is to do this (since cudaMalloc guarantees alignment):
    //    var devX = try gpu.malloc(f32, n);
    //    var devY = try gpu.malloc(f32, n);
    // But I ran into this alignment issue and I think it's better to document it here.
    const yOffset = std.mem.alignForward(usize, n, 4);
    const deviceBuffer = try gpu.malloc(f32, yOffset+n);
    defer gpu.free(deviceBuffer.ptr);
    var devX = deviceBuffer.sub(0, n);
    var devY = deviceBuffer.sub(yOffset, deviceBuffer.len);

    try devX.copyFromHost(x);
    try devY.copyFromHost(y);

    // Timing events
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);


    var params = [_]?*anyopaque{&a, @ptrCast(&devX.ptr), @ptrCast(&devY.ptr), &n};
    const threads = 256;
    var deviceProperties: cuda.cudaDeviceProp = undefined;
    try check(cuda.cudaGetDeviceProperties(&deviceProperties, gpu.device));
    const smCount: u32 = @intCast(deviceProperties.multiProcessorCount);
    { // Following the stretch goal: sweep 1–32 blocks per SM and find where bandwidth saturates.
        const saxpy_grid_stride = gpu.kernels.get("saxpy_grid_stride").?;
        // Warmup
        var config: cuda.cudaLaunchConfig_t  = .{
            .gridDim = dim3(smCount, 1, 1),
            .blockDim = dim3(threads, 1, 1),
        };
        try check(cuda.cudaLaunchKernelExC(&config, saxpy_grid_stride, &params));

        var i: u32 = 1;
        while (i <= 32) : (i *= 2) {
            config = .{
                .gridDim = dim3(i * smCount, 1, 1),
                .blockDim = dim3(threads, 1, 1),
            };

            // Reset y size it's overwritten by kernel
            try devY.copyFromHost(y);

            try check(cuda.cudaEventRecord(start, null));
            try check(cuda.cudaLaunchKernelExC(&config, saxpy_grid_stride, &params));
            try check(cuda.cudaEventRecord(end, null));
            try check(cuda.cudaDeviceSynchronize());

            var timeTaken: f32 = undefined;
            try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
            // Compute bandwidth (the 3* is for two reads and one write)
            const bandwidth = calculate_bandwidth(3 * n * @sizeOf(f32), timeTaken);
            std.debug.print("Running {} blocks per SM took {} ms ({} GB/s)\n", .{i, timeTaken, bandwidth});
        }
    }

    // Check correctness
    const out = try hostAlloc.alloc(f32, y.len);
    try devY.copyToHost(out);

    for (0..out.len) |i| {
        const expected = a * x[i] + y[i];
        if (@abs(expected - out[i]) > 0.00001) {
            std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, out.len, expected, out[i]});
            return error.ValuesDontMatch;
        }
    }

    std.debug.print("\n", .{});

    { // Second part of stretch goal: Then add a float4 version that handles the tail correctly.
        const saxpy4 = gpu.kernels.get("saxpy4").?;
        // Warmup
        var config: cuda.cudaLaunchConfig_t  = .{
            .gridDim = dim3(smCount, 1, 1),
            .blockDim = dim3(threads, 1, 1),
        };
        try check(cuda.cudaLaunchKernelExC(&config, saxpy4, &params));

        var i: u32 = 1;
        while (i <= 32) : (i *= 2) {
            config = .{
                .gridDim = dim3(i * smCount, 1, 1),
                .blockDim = dim3(threads, 1, 1),
            };

            // Reset y size it's overwritten by kernel
            try devY.copyFromHost(y);

            try check(cuda.cudaEventRecord(start, null));
            try check(cuda.cudaLaunchKernelExC(&config, saxpy4, &params));
            try check(cuda.cudaEventRecord(end, null));
            try check(cuda.cudaDeviceSynchronize());

            var timeTaken: f32 = undefined;
            try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
            // Compute bandwidth (the 3* is for two reads and one write)
            const bandwidth = calculate_bandwidth(3 * n * @sizeOf(f32), timeTaken);
            std.debug.print("(sxpy4) Running {} blocks per SM took {} ms ({} GB/s)\n", .{i, timeTaken, bandwidth});
        }
    }

    // Check correctness
    try devY.copyToHost(out);

    for (0..out.len) |i| {
        const expected = a * x[i] + y[i];
        if (@abs(expected - out[i]) > 0.00001) {
            std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, out.len, expected, out[i]});
            return error.ValuesDontMatch;
        }
    }

    std.debug.print("Results match!", .{});
}

// Bandwidth in gbs/s
fn calculate_bandwidth(bytes: usize, ms: f32) f32 {
    const bytesf: f32 = @floatFromInt(bytes);
    return (bytesf/1e9) / (ms/1000);
}

