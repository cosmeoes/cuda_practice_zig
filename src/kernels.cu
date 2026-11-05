#include <cstdint>

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

extern "C" __global__ void matadd_coalesced(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, unsigned rows, unsigned columns) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int column = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = row * columns + column;

    if (column < columns && row < rows) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void matadd_strided(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, unsigned rows, unsigned columns) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int column = threadIdx.y + blockIdx.y * blockDim.y;
    const int i = row * columns + column;

    if (column < columns && row < rows) {
        c[i] = a[i] + b[i];
    }
}


// heavy computations
__device__ uint32_t work_a(uint32_t v){ for(int k=0;k<512;++k){v^=v<<13;v^=v>>17;v^=v<<5;} return v; }
__device__ uint32_t work_b(uint32_t v){ for(int k=0;k<512;++k) v=v*1664525u+1013904223u; return v; }

extern "C" __global__ void warp_divergence_baseline(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, const unsigned length) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        out[i] = (i % 2 == 0) ? work_a(in[i]) : work_b(in[i]);
    }
}

extern "C" __global__ void warp_divergence_fixed(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, const unsigned length) {
    // Wraps 0 does the first 128 even number, wrap 1 does 128 odd number
    // Wrap 2 does the next 128 even number, wrap 3 does the next 128 even numbers etc.
    // That means that there is no warp divergence.
    const int offset = threadIdx.x / (blockDim.x / 2);
    const int pos = threadIdx.x % (blockDim.x / 2);
    const int i = pos*2 + offset + blockIdx.x * blockDim.x;

    if (i < length) {
        out[i] = (i % 2 == 0) ? work_a(in[i]) : work_b(in[i]);
    }
}

#define FULL_MASK 0xFFFFFFFF
extern "C" __global__ void warp_vote_a(const int* __restrict__ input, const unsigned length, const int threshold, int* __restrict__ result) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x*blockDim.x; i < length; i += stride) {
        int index = i + threadIdx.x;
        unsigned mask = __ballot_sync(FULL_MASK, index < length && input[index] > threshold);
        int leader = __ffs(mask) - 1;
        if (leader == threadIdx.x % warpSize) {
            int count = __popc(mask);
            atomicAdd(result, count);
        }
    }
}

extern "C" __global__ void warp_vote_b(const int* __restrict__ input, const unsigned length, int* __restrict__ output, const unsigned outputLength, const int threshold) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned mask = __ballot_sync(FULL_MASK, i < length && input[i] > threshold);
    int outputIndex = i / warpSize;
    if (threadIdx.x % warpSize == 0 && outputIndex < outputLength) {
        int first = __ffs(mask) - 1; // if not found _ffs returns 0 so the results is -1
        output[outputIndex] = first < 0 ? first : first + i;
    }
}

#define THREADS_PER_BLOCK 256
__device__ float tree_reduce(const float* __restrict__ input, const unsigned length) {
    __shared__ float shared_memory_array[THREADS_PER_BLOCK];
    const int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += stride) {
        sum += input[i];
    }

    shared_memory_array[threadIdx.x] = sum;

    for (int s = blockDim.x/2; s >= 1; s /= 2) {
        __syncthreads(); // make sure everybody wrote their sum
        if (threadIdx.x < s) {
            shared_memory_array[threadIdx.x] += shared_memory_array[threadIdx.x + s];
        }
    }

    __syncthreads(); // make sure everybody wrote their sum
    return shared_memory_array[0];
}

// Write restult with an atomicAdd
extern "C" __global__ void shared_memory_tree_reductions(const float* __restrict__ input, const unsigned length, float* __restrict__ output) {
    float sum = tree_reduce(input, length);
    if (threadIdx.x == 0) atomicAdd(output, sum);
}

// Write restult with a second pas over per thread block partials
// This is the first pass that stores per block results into partials array
extern "C" __global__ void shared_memory_tree_reductions_deterministic_fp(const float* __restrict__ input, const unsigned length, float* __restrict__ partials) {
    float sum = tree_reduce(input, length);
    // Assumes blocksDim.x <= partialsLen
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

// This is the second pass that sums partials and stores the results into the first element of the partials array
// Assumes only one block per partials[0..length]
extern "C" __global__ void shared_memory_tree_reductions_deterministic_sp(float* __restrict__ partials, const unsigned length) {
    float sum = tree_reduce(partials, length);
    if (threadIdx.x == 0) partials[0] = sum; 
}

extern "C" __global__ void shared_memory_tree_reductions_interleaved(const float* __restrict__ input, const unsigned length, float* __restrict__ output) {
    __shared__ float shared_memory_array[THREADS_PER_BLOCK];
    const int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += stride) {
        sum += input[i];
    }

    shared_memory_array[threadIdx.x] = sum;

    for (int s = 1; s < blockDim.x; s *= 2) {
        __syncthreads(); // make sure everybody wrote their sum
        if (threadIdx.x % (2*s) == 0) {
            shared_memory_array[threadIdx.x] += shared_memory_array[threadIdx.x + s];
        }
    }

    __syncthreads(); // make sure everybody wrote their sum
    if (threadIdx.x == 0) atomicAdd(output, shared_memory_array[threadIdx.x]);
}
