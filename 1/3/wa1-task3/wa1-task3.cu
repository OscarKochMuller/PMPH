#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300

void vectorAddSequential(float *A, float* B, float* C, unsigned int n) {
    for(unsigned int i = 0; i < n; ++i) {
        C[i] = A[i] +B[i];
    }
}

__global__ void vectorAddKernel(float *A, float* B, float* C, int N) {
    // compute global thread id in dimension x
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < N) { // don't access out of bounds
        C[gid] = A[gid] + B[gid];
    }
}


int main(int argc, char** argv) {

    unsigned int N;

    { // reading the number of elements
        if(argc != 2) {
            printf("Num Args is: %d instead of 1. Exiting!\n", argc);
            exit(1);
        }

        N = atoi(argv[1]);
        printf("N is: %d\n", N);

        const unsigned int maxN = 500000000;

        if(N > maxN) {
            printf("N is too big; maximal value is %d. Exiting!\n", maxN);
            exit(2);
        }
    }


    // use the first CUDA device:
    cudaSetDevice(0);


    unsigned int mem_size = N*sizeof(float);


    // allocate host memory
    float* h_A   = (float*) malloc(mem_size);
    float* h_B   = (float*) malloc(mem_size);
    float* h_cpu = (float*) malloc(mem_size);
    float* h_gpu = (float*) malloc(mem_size);


    // initialize the memory
    for(unsigned int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
    }


    struct timeval cpu_start, cpu_end, cpu_diff;
    gettimeofday(&cpu_start,NULL);
    vectorAddSequential(h_A, h_B, h_cpu, N);
    gettimeofday(&cpu_end,NULL);
    timeval_subtract(&cpu_diff, &cpu_end,&cpu_start);
    double cpu_elapsed = cpu_diff.tv_sec * 1000000 + cpu_diff.tv_usec;
    printf("CPU time: %f microseconds.\n", cpu_elapsed);


    // allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, mem_size);
    cudaMalloc((void**)&d_B, mem_size);
    cudaMalloc((void**)&d_C, mem_size);


    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);


    unsigned int B = 256; // choose a suitable block size in dimension x

    unsigned int numblocks = (N + B - 1) / B; // number of blocks in dimension x

    
    dim3 block(B, 1, 1), dim3 grid(numblocks, 1, 1); // total number of threads (numblocks*B) may overshoot N!
    


    struct timeval gpu_start, gpu_end, gpu_diff;
    gettimeofday(&gpu_start, NULL);
    for(unsigned int i = 0; i < GPU_RUNS; ++i) {
        vectorAddKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    gettimeofday(&gpu_end, NULL);
    timeval_subtract(&gpu_diff, &gpu_end, &gpu_start);
    double gpu_elapsed = gpu_diff.tv_sec * 1000000 + gpu_diff.tv_usec;
    printf("GPU time (in total): %f microseconds.\n", gpu_elapsed);
    double gpu_time = gpu_elapsed / GPU_RUNS;
    printf("GPU avg. time: %f microseconds.\n",gpu_time);
    double speedup = cpu_elapsed / gpu_time;
    printf("Speedup: %f\n", speedup);
    double throughput = (3.0 * N * sizeof(float)) / (gpu_time * 1000.0);
    printf("CUDA memory throughput: %f GB/s\n", throughput);



    // check for errors
    gpuAssert(cudaPeekAtLastError());


    // copy result from device to host
    cudaMemcpy(h_gpu, d_C, mem_size, cudaMemcpyDeviceToHost);

    const float epsilon = 0.000001f;
    int valid = 1;

    for(unsigned int i = 0; i < N; ++i) {
        if(fabs(h_cpu[i] - h_gpu[i]) >= epsilon) {
            valid = 0;
            break;
        }
    }

    if(valid) {
        printf("VALID RESULT.\n");
    }
    else {
        printf("INVALID RESULT.\n"); 
    }
    


    // clean-up memory
    free(h_A);
    free(h_B);
    free(h_cpu);
    free(h_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

    