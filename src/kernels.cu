// Kernels
extern "C" __global__ void vec_add(float* a, float* b, float* c, unsigned vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if (workIndex < vectorLength) {
        c[workIndex] = a[workIndex] + b[workIndex];
    }
}
