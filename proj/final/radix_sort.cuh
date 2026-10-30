#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

#define CUDA_CHECK(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(EXIT_FAILURE); } } while (0)
#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef QKEYS
#define QKEYS 22
#endif

constexpr int ilog2c(int x) { return x <= 1 ? 0 : 1 + ilog2c(x >> 1); }

constexpr int B = BLOCK;
constexpr int Q = QKEYS;
constexpr int LGH = ilog2c(B);   
constexpr int H = 1 << LGH;
constexpr int TILE = Q * B;
constexpr int NW = B / 32;
constexpr uint32_t PAD_KEY = 0xFFFFFFFFu;

__host__ __device__ __forceinline__ int PADQ(int i) { return i + (i >> 5); }
constexpr int TILE_SMEM = (TILE - 1)+ ((TILE - 1) >> 5) + 1;

// kernel2 cap at 48 KB
static_assert((size_t)TILE_SMEM * sizeof(uint32_t) + (size_t)(NW + 1 + 2 * H) * sizeof(int) <= 48 * 1024,
              "kernel2 48 KB static shared memory");


__device__ __forceinline__ bool bit_is_zero(uint32_t k, int bit)
{
    return bit >= 32 || ((k >> bit) & 1u) == 0u;
}


// block-wide exclusive scan;
__device__ __forceinline__ int block_exclusive_scan(int v, int* s_warp, int& total)
{
    const int lane = threadIdx.x & 31 ;
    const int wid = threadIdx.x >> 5;

    int x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[wid] =x ;
    __syncthreads();

    if (wid == 0){
        int w  = (lane < NW) ? s_warp[lane] : 0;
        int xx= w;
        #pragma unroll
        for (int o= 1; o < 32; o <<= 1) {
            int y =__shfl_up_sync(0xFFFFFFFFu, xx, o);
            if (lane>= o) xx += y;
        }


        if (lane < NW) s_warp[lane] =xx - w;
        if (lane == NW - 1) s_warp[NW]= xx ;}
    __syncthreads();

    int excl = x - v + s_warp[wid];
    total = s_warp[NW];
    __syncthreads();
    return excl;
}



constexpr int SCAN_SPT = 16;
constexpr int SCAN_TILE= SCAN_SPT * B; 
constexpr int SCAN_TILE_SMEM = (SCAN_TILE -1) + ((SCAN_TILE - 1) >> 5) + 1;

// Scan 1:  the per-tile sums
__global__ void __launch_bounds__(B)
scan_reduce_kernel(const int* __restrict__ in, int* __restrict__ block_sums, int64_t N)
{
    __shared__ int s_warp[NW + 1];
    const int64_t base = (int64_t)blockIdx.x * SCAN_TILE;

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < SCAN_SPT; i++) {

        int64_t idx = base + (int64_t)i * B + threadIdx.x;
        if (idx < N) sum += in[idx];
    }
    int total;
    block_exclusive_scan(sum, s_warp, total);

    if (threadIdx.x == 0) block_sums[blockIdx.x] =total;
}
// Scan 2:  scan the tile sums into the tile offsets
__global__ void __launch_bounds__(B)
scan_block_sums_kernel(int* __restrict__ sums, int num_sums)
{
    __shared__ int s_warp[NW + 1];

    int carry= 0;
    for (int base = 0; base <num_sums; base += B) {
        const int idx = base +threadIdx.x;
        int v= (idx < num_sums) ? sums[idx] : 0;
        int total;
        int excl= block_exclusive_scan(v, s_warp, total);
        if (idx < num_sums) sums[idx] = excl +carry;
        carry += total;
    }
}
// Scan 3:  scan each tile and add their offset
__global__ void __launch_bounds__(B)
scan_apply_kernel(const int* __restrict__ in, int* __restrict__ out,const int* __restrict__ block_offsets, int64_t N)
{
    __shared__ int tile[SCAN_TILE_SMEM];
    __shared__ int s_warp[NW + 1];

    const int tid = threadIdx.x;
    const int64_t base = (int64_t)blockIdx.x *SCAN_TILE;


    #pragma unroll
    for (int i = 0; i < SCAN_SPT; i++) {
        int64_t idx = base + (int64_t)i * B + tid ;
        tile[PADQ(i * B + tid)] = (idx < N) ? in[idx] : 0 ;
    }
    __syncthreads();

    // each thread will take contiguous elements
    int reg[SCAN_SPT];
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < SCAN_SPT; i++) {
        reg[i] = tile[PADQ(tid *SCAN_SPT + i)];
        sum += reg[i];}

    int total;
    int run = block_exclusive_scan(sum, s_warp, total) + block_offsets[blockIdx.x];

    #pragma unroll
    for (int i = 0; i < SCAN_SPT; i++) {
        tile[PADQ(tid *SCAN_SPT + i)] = run;
        run += reg[i];
    }
    __syncthreads();

    //store coalesced
    #pragma unroll
    for (int i = 0; i < SCAN_SPT;i++) {
        int64_t idx = base + (int64_t)i * B + tid;
        if (idx < N) out[idx] = tile[PADQ(i * B + tid)];
    }
}

static inline int scan_num_tiles(size_t N) { return (int)((N + SCAN_TILE - 1) / SCAN_TILE); }


static void device_exclusive_scan(const int* d_in, int* d_out, size_t N, int* d_block_sums)
{
    const int tiles = scan_num_tiles(N);
    scan_reduce_kernel <<<tiles, B>>>(d_in, d_block_sums, (int64_t)N);
    scan_block_sums_kernel <<<1, B>>>(d_block_sums, tiles);
    scan_apply_kernel <<<tiles, B>>>(d_in, d_out, d_block_sums, (int64_t)N);
}

// Per block digit histogram
__global__ void __launch_bounds__(B)
kernel1_histogram(const uint32_t* __restrict__ val_in, int* __restrict__ hist_2D,int n, int bit_offset)
{
    __shared__ int shared_histogram[H];
    const int tid = threadIdx.x;
    const int64_t tile_start = (int64_t)blockIdx.x * TILE;

    for (int k = tid; k < H; k += B) shared_histogram[k] = 0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int64_t idx = tile_start + (int64_t)i * B + tid;
        if (idx < n) {
            uint32_t digit = (val_in[idx] >> bit_offset) & (H - 1);
            atomicAdd(&shared_histogram[digit], 1);
        }}
    __syncthreads();

    for (int k = tid; k < H; k += B)
        hist_2D[blockIdx.x * H + k] =shared_histogram[k];
}

__global__ void transpose_kernel(const int* __restrict__ in, int* __restrict__ out, int rows, int cols)
{
    __shared__ int t[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < cols && y + j < rows) t[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < rows && y + j < cols) out[(size_t)(y + j) * rows + x] = t[threadIdx.x][threadIdx.y+ j];
}

// Kernel 2: block-local partition + scatter
__global__ void __launch_bounds__(B)
kernel2_partition_scatter(const uint32_t* __restrict__ val_in, uint32_t* __restrict__ val_out,const int* __restrict__ hist_2D, const int* __restrict__ scanned_T,int n, int bit_offset, int num_blocks)
{
    __shared__ uint32_t tile[TILE_SMEM];
    __shared__ int s_warp[NW + 1];
    __shared__ int local_hist[H];
    __shared__ int global_off[H];

    const int tid = threadIdx.x;
    const int64_t tile_start = (int64_t)blockIdx.x * TILE;
    const int valid = (int)min((int64_t)TILE, (int64_t)n - tile_start);

    uint32_t reg[Q];

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int64_t idx = tile_start + (int64_t)i * B + tid;
        tile[PADQ(i * B + tid)] = (idx < n) ? val_in[idx] : PAD_KEY;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) reg[i] = tile[PADQ(tid * Q + i)];
    __syncthreads();

    #pragma unroll 1
    for (int b = 0; b < LGH; b++) {
        const int bit = bit_offset + b;

        int cnt = 0;
        #pragma unroll
        for (int i = 0; i < Q; i++) cnt += bit_is_zero(reg[i], bit);

        int total_true;
        int run_true = block_exclusive_scan(cnt, s_warp, total_true);
        int run_false = tid * Q - run_true;

        #pragma unroll
        for (int i = 0; i < Q; i++) {
            bool p = bit_is_zero(reg[i], bit);
            int dest;
            if (p) { dest = run_true;run_true++;}
            else   { dest = total_true + run_false; run_false++; }
            tile[PADQ(dest)] = reg[i];
        }
        __syncthreads();

        if (b < LGH -1) {
            #pragma unroll
            for (int i = 0; i < Q;i++) reg[i] = tile[PADQ(tid * Q + i)];
            __syncthreads();
        }}
{
        int h = hist_2D[blockIdx.x * H + tid];
        int dummy;
        local_hist[tid] = block_exclusive_scan(h, s_warp, dummy);
        global_off[tid] = scanned_T[(size_t)tid * num_blocks + blockIdx.x];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i <Q; i++) {
        int pos = i * B + tid;
        if (pos < valid) {
            uint32_t key =tile[PADQ(pos)];
            int d = (key >>bit_offset) & (H - 1);
            val_out[global_off[d] + (pos - local_hist[d])] = key;
        }}}

// Host kernel

void radix_sort(uint32_t* d_in, uint32_t* d_out, int n)
{
    if (n <= 1) { if (n==1) CUDA_CHECK(cudaMemcpy(d_out, d_in, 4, cudaMemcpyDeviceToDevice)); return; }

    const int num_blocks = (n + TILE - 1) / TILE;
    const size_t hist_sz = (size_t)num_blocks * H;

    int *hist_2D, *hist_T, *scanned_T;
    CUDA_CHECK(cudaMalloc(&hist_2D,   hist_sz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hist_T,    hist_sz *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scanned_T, hist_sz * sizeof(int)));

    int* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, (size_t)scan_num_tiles(hist_sz) * sizeof(int)));

    dim3 tr_block(32, 8), tr_grid(H / 32, (num_blocks + 31) / 32);

    uint32_t* src = d_in;
    uint32_t* dst = d_out;
    for (int bit_offset = 0; bit_offset < 32; bit_offset += LGH) {
        kernel1_histogram<<<num_blocks, B>>>(src, hist_2D, n, bit_offset);
        transpose_kernel<<<tr_grid, tr_block>>>(hist_2D,hist_T, num_blocks, H);
        device_exclusive_scan(hist_T, scanned_T, hist_sz, d_block_sums);
        kernel2_partition_scatter<<<num_blocks, B>>>(src, dst, hist_2D, scanned_T,n, bit_offset, num_blocks);
        std::swap(src, dst);
    }
    CUDA_CHECK(cudaMemcpy(d_out, src, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(hist_2D));
    CUDA_CHECK(cudaFree(hist_T));
    CUDA_CHECK(cudaFree(scanned_T));
    CUDA_CHECK(cudaFree(d_block_sums));
}
