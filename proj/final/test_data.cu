#include "radix_sort.cuh"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

static const char* names[] = {"random", "sorted", "reversed", "all-zero", "all-max", "few-distinct"};
static const int NPAT   = 6;
static const int WARMUP = 2;
static const int REPS   = 20;

static void fill_pattern(int p, std::vector<uint32_t>& d, std::mt19937& rng)
{
    for (auto& x : d) {
        switch (p) {
            case 0: x = rng(); break;
            case 1: x = rng(); break;                  
            case 2: x = rng(); break;                  
            case 3: x = 0u; break;
            case 4: x = 0xFFFFFFFFu; break;
            case 5: x = (rng() % 4) * 0x55555555u; break;
        }
    }
    if (p == 1) std::sort(d.begin(), d.end());
    if (p == 2) std::sort(d.begin(), d.end(), std::greater<uint32_t>());
}


static float bench_one(int p, int n, std::mt19937& rng,
                       uint32_t* d_src, uint32_t* d_in, uint32_t* d_out,
                       cudaEvent_t t0, cudaEvent_t t1, bool* ok)
{
    std::vector<uint32_t> d(n), ref, res(n);
    fill_pattern(p, d, rng);
    ref = d;
    std::sort(ref.begin(), ref.end());
    CUDA_CHECK(cudaMemcpy(d_src, d.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

    // checked run
    CUDA_CHECK(cudaMemcpy(d_in, d_src, (size_t)n * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xA5, (size_t)n * 4));
    radix_sort(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(res.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
    *ok = (res == ref);

    // warm-up
    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpy(d_in, d_src, (size_t)n * 4, cudaMemcpyDeviceToDevice));
        radix_sort(d_in, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed run
    double sum = 0.0;
    for (int i = 0; i < REPS; i++) {
        CUDA_CHECK(cudaMemcpy(d_in, d_src, (size_t)n * 4, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        radix_sort(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum += ms;
    }
    return (float)(sum / REPS);
}

int main()
{
    
    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    const int NS = sizeof(sizes) / sizeof(sizes[0]);
    const int NMAX = sizes[NS - 1];

    uint32_t *d_src, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_src, (size_t)NMAX * 4));
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)NMAX * 4));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)NMAX * 4));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    std::mt19937 rng(12345);
    float avg[NPAT][8];
    int failures = 0;

    for (int s = 0; s < NS; s++)
        for (int p = 0; p < NPAT; p++) {
            bool ok;
            avg[p][s] = bench_one(p, sizes[s], rng, d_src, d_in, d_out, t0, t1, &ok);
            if (!ok) { failures++; printf("WRONG RESULT: %s n=%d\n", names[p], sizes[s]); }
        }

    printf("Average time [ms] over %d runs\n%-14s", REPS, "pattern");
    for (int s = 0; s < NS; s++) printf(" %12d", sizes[s]);
    printf("\n");
    for (int p = 0; p < NPAT; p++) {
        printf("%-14s", names[p]);
        for (int s = 0; s < NS; s++) printf(" %12.4f", avg[p][s]);
        printf("\n");
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return failures ? 1 : 0;
}
