// Kernels
extern "C" __global__ void vec_add(float* a, float* b, float* c, unsigned vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if (workIndex < vectorLength) {
        c[workIndex] = a[workIndex] + b[workIndex];
    }
}

extern "C" __global__ void saxpy_grid_stride(float a, const float* __restrict__ x, float* __restrict__ y, unsigned n) {
    const int stride = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void saxpy4(float a, const float* __restrict__ x, float* __restrict__ y, unsigned n) {
    const int stride = gridDim.x * blockDim.x;

    const unsigned n4 = n / 4;
    const float4* x4 = (float4*) x;
    float4* y4 = (float4*) y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n4; i += stride) {
        float4 xv = x4[i], yv = y4[i];
        yv.x = a * xv.x + yv.x;
        yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z;
        yv.w = a * xv.w + yv.w;
        y4[i] = yv;
    }

    for (int i = n4 * 4 + tid; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}
