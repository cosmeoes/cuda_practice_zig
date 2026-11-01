const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const cuda = @import("cuda");

const ptx: [*c]const u8 = @embedFile("kernels.ptx");

const GPU = struct {
    device: cuda.CUdevice = 0,
    context: cuda.CUcontext = null,
    module: cuda.CUmodule = null,
    // Functions
    f: std.StringHashMap(cuda.CUfunction),

    pub fn init(alloc: std.mem.Allocator) !GPU {
        var gpu: GPU = .{ .f = .init(alloc) };
        try check(cuda.cuInit(0));
        try check(cuda.cuDeviceGet(&gpu.device, 0));
        try check(cuda.cuCtxCreate(&gpu.context, null, 0, gpu.device));
        try check(cuda.cuModuleLoadData(&gpu.module, ptx));

        // Load functions
        var count: c_uint = 0;
        try check(cuda.cuModuleGetFunctionCount(&count, gpu.module));
        const functions: []cuda.CUfunction = try alloc.alloc(cuda.CUfunction, @intCast(count)); // @leak
        try check(cuda.cuModuleEnumerateFunctions(@ptrCast(functions.ptr), count, gpu.module));
        for (functions) |f| {
            var name: [*c]const u8 = null;
            try check(cuda.cuFuncGetName(&name, f));
            try gpu.f.put(std.mem.span(name), f);
            std.debug.print("{s}\n", .{name});
        }

        return gpu;
    }
};

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    const gpu: GPU = try .init(arena);
    try unified_memory_example(io, gpu, 1024);
}

pub fn unified_memory_example(io: std.Io, gpu: GPU, vectorLength: c_uint) !void {
    var a: []f32 = undefined;
    var b: []f32 = undefined;
    var c: []f32 = undefined;
    a.len = @intCast(vectorLength);
    b.len = @intCast(vectorLength);
    c.len = @intCast(vectorLength);
    try check(cuda.cudaMallocManaged(@ptrCast(&a.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cuMemFree(@intFromPtr(a.ptr));
    try check(cuda.cudaMallocManaged(@ptrCast(&b.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cuMemFree(@intFromPtr(b.ptr));
    try check(cuda.cudaMallocManaged(@ptrCast(&c.ptr), vectorLength*@sizeOf(f32), cuda.cudaMemAttachGlobal));
    defer _ = cuda.cuMemFree(@intFromPtr(c.ptr));

    init_array(io, a);
    init_array(io, b);

    const threads = 265;
    const config: cuda.CUlaunchConfig = .{
        .gridDimX = @divCeil(vectorLength, threads),
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = threads,
        .blockDimY = 1,
        .blockDimZ = 1,
    };

    // Launch
    var params = [_]?*anyopaque{@ptrCast(&a.ptr), @ptrCast(&b.ptr), @ptrCast(&c.ptr), @constCast(&vectorLength)};
    const vec_add = gpu.f.get("vec_add").?;
    try check(config.cuLaunchKernelEx(vec_add, @ptrCast(&params), null));
    try check(cuda.cudaDeviceSynchronize());

    assert(c.len > 0);
    for (0..c.len) |i| {
        std.debug.print("{d} + {d} = {d}\n", .{a[i], b[i], c[i]});
    }
}

fn init_array(io: std.Io,  a: []f32) void {
    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(io).toMilliseconds()));
    const random = prng.random();
    for (a) |*x| {
        x.* = random.float(f32);
    }
}

fn check(r: cuda.CUresult) !void {
    if (r != cuda.CUDA_SUCCESS) {
        var name: [*c]const u8 = null;
        _ = cuda.cuGetErrorName(r, &name);
        std.debug.print("CUDA error: {s}\n", .{name});
        return error.CudaError;
    }
}
