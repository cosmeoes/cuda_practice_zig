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

        pub fn memset(self: @This(), value: i32) !void {
            try check(cuda.cudaMemset(self.ptr, value, self.byteSize()));
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
    // try unified_memory_example(io, gpu, 1024);
    // try explicit_mem_example(io, &gpu, 1024);
    // try saxpy_grid_stride_problem(io, &gpu);
    // try matadd_problem(io, &gpu);
    // try warp_divergence_problem(io, &gpu);
    // try warp_vote_problem_a(io, &gpu);
    // try warp_vote_problem_b(io, &gpu);
    try shared_memory_tree_reductions_problem(io, &gpu);
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

// Bandwidth in gbs/s
fn calculate_bandwidth(bytes: usize, ms: f32) f32 {
    const bytesf: f32 = @floatFromInt(bytes);
    return (bytesf/1e9) / (ms/1000);
}

fn init_array(random: std.Random,  a: []f32) void {
    for (0..a.len) |i| {
        a[i] = random.float(f32);
    }
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

// Compute C = A + B for a row-major 3001 × 4099 float matrix, using 2D blocks of (32, 8). Write two kernels:
// - coalesced: the x dimension (threadIdx.x / blockIdx.x) walks columns.
// - strided: the x dimension walks rows.
// 
// Use one thread per element: every element of the matrix gets exactly one thread, with no grid-stride loop. 
// Choosing each kernel's grid size is part of the exercise, and it depends on which matrix dimension that kernel maps to x. Guard against out-of-range indices.
// Measure: the bandwidth of both. Explain the gap in terms of how the 32 lanes of one warp map to memory addresses.
// Stretch: in Nsight Compute, compare "sectors per request" between the two kernels.
pub fn matadd_problem(io: std.Io, gpu: *GPU) !void {
    //  Messuraments on my 3070 ti:
    //    matadd_coalesced took 0.275416 ms (535.9645 GB/s)
    //    matadd_strided took 0.8248416 ms (178.95943 GB/s)
    // The thread blocks have a width of 32 so that means that each row of the thread blocks corresponds
    // to a warp. In the matadd_coalesced the reads/writes are sequencial within each warp.
    // t1 reads/writes offset i, t2 reads/writes offset i+1, t3 reads/writes offset i+2 .. t32 reads/writes offset i+31
    // So for each read/write (a and b are read and c is written) the warp reads/writes 128 bytes
    // which can be executed in 4 transactions (best case scenario, worse is 5) since it only touches 4 sections in global memory (sections are 32 bytes).
    // In the metadd_strided the access patter looks more like:
    // t1 reads/writes offset i, t2 reads/writes offset i + columns, t3 reads/writes offset i + columns*2 .. t32 reads/writes offset i + columns*31
    // Since the reads are no sequencial each read/write touches a different section which requires 32 transactions and it fetches 1024 bytes and 
    // only uses the same 128 bytes which increases memory traffic unnecessarily. (Although cache helps)

    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var rows: u32 = 3001;
    var cols: u32 = 4099;
    const size = rows*cols;

    const hostBuffer = try hostAlloc.alloc(f32, size*3);
    const a = hostBuffer[0..size];
    const b = hostBuffer[size..size*2];
    const c = hostBuffer[size*2..hostBuffer.len];

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    init_array(random, a);
    init_array(random, b);

    const deviceBuffer = try gpu.malloc(f32, rows*cols * 3);
    defer gpu.free(deviceBuffer.ptr);
    var devA = deviceBuffer.sub(0, rows*cols);
    var devB = deviceBuffer.sub(rows*cols, rows*cols*2);
    var devC = deviceBuffer.sub(rows*cols*2, deviceBuffer.len);

    try devA.copyFromHost(a);
    try devB.copyFromHost(b);

    // Timing events
    const sampleSize = 20;
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);

    var params = [_]?*anyopaque{@ptrCast(&devA.ptr), @ptrCast(&devB.ptr), @ptrCast(&devC.ptr), &rows, &cols};
    // matadd_coallesced
    {
        const matadd_coalesced = gpu.kernels.get("matadd_coalesced").?;
        const blockDim = dim3(32, 8, 1);
        const gridDim = dim3(@divCeil(cols, blockDim.x), @divCeil(rows, blockDim.y), 1);
        var config: cuda.cudaLaunchConfig_t  = .{
            .gridDim = gridDim,
            .blockDim = blockDim,
        };
        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaLaunchKernelExC(&config, matadd_coalesced, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;
        // Compute bandwidth (the 3* is for two reads and one write)
        const bandwidth = calculate_bandwidth(3 * size * @sizeOf(f32), timeTaken);
        std.debug.print("matadd_coalesced took {} ms ({} GB/s)\n", .{timeTaken, bandwidth});

        // Check correctness 
        try devC.copyToHost(c);
        for (0..c.len) |i| {
            const expected = a[i] + b[i];
            if (@abs(expected - c[i]) > 0.00001) {
                std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, c.len, expected, c[i]});
                return error.ValuesDontMatch;
            }
        }
    }

    // Reset the output to make sure it doesn't influence the next kernel
    try devC.memset(0xFF);

    // matadd_strided
    {
        const matadd_strided = gpu.kernels.get("matadd_strided").?;
        const blockDim = dim3(32, 8, 1);
        const gridDim = dim3(@divCeil(rows, blockDim.x), @divCeil(cols, blockDim.y), 1);
        var config: cuda.cudaLaunchConfig_t  = .{
            .gridDim = gridDim,
            .blockDim = blockDim,
        };
        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaLaunchKernelExC(&config, matadd_strided, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;
        // Compute bandwidth (the 3* is for two reads and one write)
        const bandwidth = calculate_bandwidth(3 * size * @sizeOf(f32), timeTaken);
        std.debug.print("matadd_strided took {} ms ({} GB/s)\n", .{timeTaken, bandwidth});

        // Check correctness 
        try devC.copyToHost(c);
        for (0..c.len) |i| {
            const expected = a[i] + b[i];
            if (@abs(expected - c[i]) > 0.00001) {
                std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, c.len, expected, c[i]});
                return error.ValuesDontMatch;
            }
        }
    }

    std.debug.print("Results match!", .{});
}

// Two expensive pure functions on uint32_t (each runs 512 iterations):
//    __host__ __device__ uint32_t work_a(uint32_t v){ for(int k=0;k<512;++k){v^=v<<13;v^=v>>17;v^=v<<5;} return v; }
//    __host__ __device__ uint32_t work_b(uint32_t v){ for(int k=0;k<512;++k) v=v*1664525u+1013904223u; return v; }
// The baseline kernel computes out[i] = (i % 2 == 0) ? work_a(in[i]) : work_b(in[i]), with one thread per element.
// Task: write the baseline, then write a second kernel that produces bit-identical output where every branch is uniform across each warp. Leave the per-element work alone; change which thread handles which element.
// Launch configuration, the same for both kernels: n = 2^22, blocks of 256 threads, one thread per element. Block b owns elements [256·b, 256·b + 256). 
// In the second kernel, a block can reassign the elements of its own chunk among its threads however it likes, but it can't touch elements outside its chunk.
// Measure: the speed-up (expect about 2×).
// Stretch: change the baseline's condition to i % 64 < 32 and predict the timing before you run it.
pub fn warp_divergence_problem(io: std.Io, gpu: *GPU) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var n: u32 = std.math.pow(u32, 2, 22);

    const hostBuffer = try hostAlloc.alloc(u32, n*2);
    const in = hostBuffer[0..n];
    const out = hostBuffer[n..hostBuffer.len];

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    for (in) |*x| {
        x.* = random.int(u32);
    }

    const deviceBuffer = try gpu.malloc(u32, n*2);
    defer gpu.free(deviceBuffer.ptr);
    var devIn = deviceBuffer.sub(0, n);
    var devOut = deviceBuffer.sub(n, deviceBuffer.len);

    try devIn.copyFromHost(in);

    // Timing events
    const sampleSize = 20;
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);

    var params = [_]?*anyopaque{@ptrCast(&devIn.ptr), @ptrCast(&devOut.ptr), &n};
    var config: cuda.cudaLaunchConfig_t  = .{
        .gridDim = dim3(@divCeil(n, 256), 1, 1),
        .blockDim = dim3(256, 1, 1),
    };

    const f = struct {
        fn work_a(v_in: u32) u32 {
            var v = v_in;
            var k: u32 = 0;
            while (k < 512) : (k+=1) {
                v ^= v<<13;
                v ^= v>>17;
                v ^= v<<5;
            }
            return v;
        }

        fn work_b(v_in: u32) u32 {
            var v = v_in;
            var k: u32 = 0;
            while (k < 512) : (k+=1) {
                v = v *% 1664525 +% 1013904223;
            }
            return v;
        }
    };

    { // Baseline
        const warp_divergence_baseline = gpu.kernels.get("warp_divergence_baseline").?;
        // warmup
        try check(cuda.cudaLaunchKernelExC(&config, warp_divergence_baseline, &params));

        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaLaunchKernelExC(&config, warp_divergence_baseline, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        try devOut.copyToHost(out);

        for (0..out.len) |i| {
            const expected = if (i % 2 == 0) f.work_a(in[i]) else f.work_b(in[i]);
            if (expected != out[i]) {
                std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, out.len, expected, out[i]});
                return error.ValuesDontMatch;
            }
        }

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;

        // Compute bandwidth (the 2 * is for one read and one write)
        const bandwidth = calculate_bandwidth(2 * n * @sizeOf(f32), timeTaken);
        std.debug.print("wrap_divergence_baseline took {} ms ({} GB/s)\n", .{timeTaken, bandwidth});
    }

    // Clear output for next kernel
    try devOut.memset(0x0F);

    { // Fixed
        const warp_divergence_fixed = gpu.kernels.get("warp_divergence_fixed").?;
        // warmup
        try check(cuda.cudaLaunchKernelExC(&config, warp_divergence_fixed, &params));

        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaLaunchKernelExC(&config, warp_divergence_fixed, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        try devOut.copyToHost(out);

        for (0..out.len) |i| {
            const expected = if (i % 2 == 0) f.work_a(in[i]) else f.work_b(in[i]);
            if (expected != out[i]) {
                std.debug.print("Index {} of {} mismatch: {} != {}\n", .{i, out.len, expected, out[i]});
                return error.ValuesDontMatch;
            }
        }

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;

        // Compute bandwidth (the 2 * is for one read and one write)
        const bandwidth = calculate_bandwidth(2 * n * @sizeOf(u32), timeTaken);
        std.debug.print("wrap_divergence_fixed took {} ms ({} GB/s)\n", .{timeTaken, bandwidth});
    }
}

// The input is n = 10,000,019 ints in [0, 999], with threshold thr = 990.
// A. Count the elements where data[i] > thr. Use a grid-stride loop over a fixed small grid, and do one atomic per warp per iteration,
// not one per thread. Watch out: every lane in the ballot mask has to reach the __ballot_sync. 
// A naive grid-stride loop lets some lanes leave early on the last iteration.
pub fn warp_vote_problem_a(io: std.Io, gpu: *GPU) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var n: u32 = 10_000_019;
    var threshold: i32 = 990;

    const input = try hostAlloc.alloc(i32, n);

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const random = prng.random();
    var expectedResult: u32 = 0;
    for (input) |*x| {
        x.* = random.intRangeLessThan(i32, 0, 1_000);
        if (x.* > threshold) {
            expectedResult += 1;
        }
    }

    var devInput = try gpu.malloc(i32, n);
    defer gpu.free(devInput.ptr);
    var devResult = try gpu.malloc(i32, 1);
    defer gpu.free(devResult.ptr);
    try devResult.memset(0);

    try devInput.copyFromHost(input);

    // Timing events
    const sampleSize = 20;
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);

    var params = [_]?*anyopaque{@ptrCast(&devInput.ptr), &n, &threshold, @ptrCast(&devResult.ptr)};
    var config: cuda.cudaLaunchConfig_t  = .{
        .gridDim = dim3(100, 1, 1),
        .blockDim = dim3(256, 1, 1),
    };

    const warp_vote_a = gpu.kernels.get("warp_vote_a").?;
    // warmup
    try check(cuda.cudaLaunchKernelExC(&config, warp_vote_a, &params));

    try check(cuda.cudaEventRecord(start, null));
    for (0..sampleSize) |_| {
        try devResult.memset(0);
        try check(cuda.cudaLaunchKernelExC(&config, warp_vote_a, &params));
    }
    try check(cuda.cudaEventRecord(end, null));
    try check(cuda.cudaDeviceSynchronize());

    var result: i32 = 0;
    try check(cuda.cudaMemcpy(&result, devResult.ptr, @sizeOf(i32), cuda.cudaMemcpyDeviceToHost));

    if (expectedResult != result) {
        std.debug.print("Result doesn't match with expected result. Expected {}. Got {}\n", .{expectedResult, result});
    }

    var timeTaken: f32 = undefined;
    try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
    timeTaken = timeTaken / sampleSize;
    const bandwidth = calculate_bandwidth(n * @sizeOf(u32), timeTaken);
    std.debug.print("warp_vote took {} ms ({} GB/s) \n", .{timeTaken, bandwidth});
}
// Continuation of prev problem:
// B. First match per chunk: split the data into chunks of 32 elements. For each chunk, output the global index of the first element > thr, 
// or -1 if there isn't one. Use one thread per element, so one warp handles one chunk.
pub fn warp_vote_problem_b(io: std.Io, gpu: *GPU) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var n: u32 = 10_000_019;
    var groupCount: u32 = @divCeil(n, 32);
    var threshold: i32 = 990;

    const input = try hostAlloc.alloc(i32, n);
    const output = try hostAlloc.alloc(i32, groupCount);
    const expected = try hostAlloc.alloc(i32, groupCount);
    @memset(output, 0);
    @memset(expected, 0);

    const seed: u64 = @intCast(std.Io.Clock.real.now(io).toMicroseconds());
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (input) |*x| {
        x.* = random.intRangeLessThan(i32, 0, 1_000);
    }

    { // Calculate results in cpu to verify GPU result
        var i: usize = 0;
        while (i < input.len) : (i += 32) {
            var chunkResult: i32 = -1;
            for (i..i+32) |j| {
                if (j < input.len and input[j] > threshold) {
                    chunkResult = @intCast(j);
                    break;
                }
            }
            expected[@divFloor(i, 32)] = chunkResult;
        }
    }

    var devInput = try gpu.malloc(i32, n);
    defer gpu.free(devInput.ptr);
    var devOutput = try gpu.malloc(i32, groupCount);
    defer gpu.free(devOutput.ptr);

    try devInput.copyFromHost(input);

    // Timing events
    const sampleSize = 20;
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);

    var params = [_]?*anyopaque{@ptrCast(&devInput.ptr), &n, @ptrCast(&devOutput.ptr), &groupCount, &threshold};
    const threads = 256;
    var config: cuda.cudaLaunchConfig_t  = .{
        .gridDim = dim3(@divCeil(n, threads), 1, 1),
        .blockDim = dim3(256, 1, 1),
    };

    const warp_vote_b = gpu.kernels.get("warp_vote_b").?;
    // warmup
    try check(cuda.cudaLaunchKernelExC(&config, warp_vote_b, &params));

    try check(cuda.cudaEventRecord(start, null));
    for (0..sampleSize) |_| {
        try check(cuda.cudaLaunchKernelExC(&config, warp_vote_b, &params));
    }
    try check(cuda.cudaEventRecord(end, null));
    try check(cuda.cudaDeviceSynchronize());

    try devOutput.copyToHost(output);

    // Verify correctness
    for (0..output.len) |i| {
        if (expected[i] != output[i]) {
            std.debug.print("Index {} mismatch: {} != {}\n", .{i, expected[i], output[i]});
            return error.ValueDoesntMatch;
        }
    }

    var timeTaken: f32 = undefined;
    try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
    timeTaken = timeTaken / sampleSize;
    const bandwidth = calculate_bandwidth(n * @sizeOf(i32) + output.len * @sizeOf(i32), timeTaken);
    std.debug.print("warp_vote took {} ms ({} GB/s) \n", .{timeTaken, bandwidth});
}

// Sum n = 2^25 + 3 floats in [0, 1]. Each block should:
// 1. accumulate per-thread partial sums with a grid-stride loop,
// 2. reduce them in __shared__ memory with a tree using sequential addressing (s = BLOCK/2 … 1, sdata[tid] += sdata[tid+s]),
// 3. atomicAdd its result into a single global float.
// Check against a CPU sum done in double, with a relative tolerance of 1e-4.
// Stretch: write the "interleaved addressing" version (tid % (2*s) == 0) and measure it against yours.
// Then make the result deterministic by replacing the atomic with a second pass over per-block partials.
pub fn shared_memory_tree_reductions_problem(io: std.Io, gpu: *GPU) !void {
    var hostArena = std.heap.ArenaAllocator.init(gpu.host_allocator.allocator());
    defer _ = hostArena.deinit();
    var hostAlloc = hostArena.allocator();

    var n = std.math.pow(u32, 2, 25);

    const input = try hostAlloc.alloc(f32, n);

    const seed = 0;
    // const seed: u64 = @intCast(std.Io.Clock.real.now(io).toMicroseconds());
    _ = io;
    var prng = std.Random.DefaultPrng.init(seed);
    // This generates numbers in [0, 1). But I think that's good enough for this exercise
    init_array(prng.random(), input);

    // Generate expected results
    var expected: f64 = 0; // instructions specify a double for cpu sum
    for (input) |i| {
        expected += @floatCast(i);
    }

    var devInput = try gpu.malloc(f32, n);
    defer gpu.free(devInput.ptr);
    var devOut: *f32 = @ptrCast((try gpu.malloc(f32, 1)).ptr);
    defer gpu.free(devOut);
    try check(cuda.cudaMemset(devOut, 0, @sizeOf(f32)));

    try devInput.copyFromHost(input);

    // Timing events
    const sampleSize = 20;
    var start: cuda.cudaEvent_t = undefined;
    var end: cuda.cudaEvent_t = undefined;
    try check(cuda.cudaEventCreate(&start));
    defer _ = cuda.cudaEventDestroy(start);
    try check(cuda.cudaEventCreate(&end));
    defer _ = cuda.cudaEventDestroy(end);

    var params = [_]?*anyopaque{@ptrCast(&devInput.ptr), &n, @ptrCast(&devOut)};
    const threads = 256;
    var config: cuda.cudaLaunchConfig_t = .{
        .gridDim = dim3(@divCeil(n, threads), 1, 1),
        .blockDim = dim3(threads, 1, 1),
    };

    {
        const shared_memory_tree_reduction = gpu.kernels.get("shared_memory_tree_reductions").?;
        // warm up
        try check(cuda.cudaLaunchKernelExC(&config, shared_memory_tree_reduction, &params));

        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaMemset(devOut, 0, @sizeOf(f32)));
            try check(cuda.cudaLaunchKernelExC(&config, shared_memory_tree_reduction, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        var out: f32 = 0.0;
        try check(cuda.cudaMemcpy(&out, devOut, @sizeOf(f32), cuda.cudaMemcpyDeviceToHost));

        // Verify correctness
        if (!std.math.approxEqRel(f64, expected, @floatCast(out), 1e-4)) {
            std.debug.print("Result doesn't match with expected result. Expected {}. Got {}\n", .{expected, out});
        } 

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;
        const bandwidth = calculate_bandwidth(n * @sizeOf(f32), timeTaken); 
        std.debug.print("shared_memory_tree_reductions (sequential) took {} ms ({} GB/s) \n", .{timeTaken, bandwidth});
    }
    {
        // This version is slower because of higher wrap divergence
        const shared_memory_tree_reduction_interleaved = gpu.kernels.get("shared_memory_tree_reductions_interleaved").?;
        // warm up
        try check(cuda.cudaLaunchKernelExC(&config, shared_memory_tree_reduction_interleaved, &params));

        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaMemset(devOut, 0, @sizeOf(f32)));
            try check(cuda.cudaLaunchKernelExC(&config, shared_memory_tree_reduction_interleaved, &params));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        var out: f32 = 0.0;
        try check(cuda.cudaMemcpy(&out, devOut, @sizeOf(f32), cuda.cudaMemcpyDeviceToHost));

        // Verify correctness
        if (!std.math.approxEqRel(f64, expected, @floatCast(out), 1e-4)) {
            std.debug.print("Result doesn't match with expected result. Expected {}. Got {}\n", .{expected, out});
        }

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;
        const bandwidth = calculate_bandwidth(n * @sizeOf(f32), timeTaken); 
        std.debug.print("shared_memory_tree_reductions (interleaved) took {} ms ({} GB/s) \n", .{timeTaken, bandwidth});
    }
    { // using two passes to make it deterministic
        var devPartials = try gpu.malloc(f32, config.gridDim.x); // one partial per block
        defer gpu.free(devPartials.ptr);

        const first_pass = gpu.kernels.get("shared_memory_tree_reductions_deterministic_fp").?;
        var firstPassParameters = [_]?*anyopaque{@ptrCast(&devInput.ptr), &n, @ptrCast(&devPartials.ptr)};
        const second_pass = gpu.kernels.get("shared_memory_tree_reductions_deterministic_sp").?;
        var len: u32 = @intCast(devPartials.len);
        var secondPassParameters = [_]?*anyopaque{@ptrCast(&devPartials.ptr), &len};

        try check(cuda.cudaEventRecord(start, null));
        for (0..sampleSize) |_| {
            try check(cuda.cudaLaunchKernelExC(&config, first_pass, &firstPassParameters));
            try check(cuda.cudaLaunchKernelExC(&.{ 
                .gridDim = dim3(1, 1, 1), .blockDim = config.blockDim ,
            }, second_pass, &secondPassParameters));
        }
        try check(cuda.cudaEventRecord(end, null));
        try check(cuda.cudaDeviceSynchronize());

        var out: f32 = 0.0;
        try check(cuda.cudaMemcpy(&out, devPartials.ptr, @sizeOf(f32), cuda.cudaMemcpyDeviceToHost));

        // Verify correctness
        if (!std.math.approxEqRel(f64, expected, @floatCast(out), 1e-4)) {
            std.debug.print("Result doesn't match with expected result. Expected {}. Got {}\n", .{expected, out});
        } 

        var timeTaken: f32 = undefined;
        try check(cuda.cudaEventElapsedTime(&timeTaken, start, end));
        timeTaken = timeTaken / sampleSize;
        const bandwidth = calculate_bandwidth(n * @sizeOf(f32) + devPartials.len * @sizeOf(f32), timeTaken); 
        std.debug.print("shared_memory_tree_reductions (deterministic) took {} ms ({} GB/s) \n", .{timeTaken, bandwidth});
    }
}

