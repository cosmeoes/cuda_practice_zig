const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build kernel ptx
    const nvcc_compile = b.addSystemCommand(&.{"nvcc", "-ptx", "-arch=native", "src/kernels.cu", "-o"});
    nvcc_compile.addFileInput(b.path("src/kernels.cu"));
    const nvcc_output = nvcc_compile.addOutputFileArg2("kernels.ptx", .{});

    // Copy ptx to zig-out for debugging
    const install_ptx = b.addInstallFile(nvcc_output, "kernels.ptx");
    b.getInstallStep().dependOn(&install_ptx.step);

    const cuda_translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/cuda.h"),
        .target = target,
        .optimize = optimize,
    });

    cuda_translate_c.linkSystemLibrary("cuda", .{});
    cuda_translate_c.linkSystemLibrary("cudart", .{});

    const exe = b.addExecutable(.{
        .name = "cuda_practice_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "cuda", .module = cuda_translate_c.createModule()},
            },
        }),
    });
    exe.step.dependOn(&nvcc_compile.step);
    exe.root_module.addAnonymousImport("kernels.ptx", .{ .root_source_file = nvcc_output });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    run_cmd.addPassthruArgs();
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
}
