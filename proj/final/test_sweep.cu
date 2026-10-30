#include "radix_sort.cuh"

static float time_ms(cudaEvent_t a,cudaEvent_t b) {float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

int main()
{
    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    const int RUNS = 10;

    printf("CONFIG: B=%d Q=%d LGH=%d passes=%d TILE=%d\n", B, Q, LGH, (32 + LGH - 1) / LGH, TILE);

    std::mt19937 rng(12345);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int n : sizes) {
        // RUNS 
        std::vector<std::vector<uint32_t>> data(RUNS), refs(RUNS);
        for (int r = 0; r < RUNS; r++) {
            data[r].resize(n);
            for (auto& x : data[r]) x = rng();
            refs[r] = data[r];
            std::sort(refs[r].begin(), refs[r].end());
        }

        uint32_t *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * 4));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * 4));

        std::vector<uint32_t> res(n);

        // warm-up 
        CUDA_CHECK(cudaMemcpy(d_in, data[0].data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        radix_sort(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // timed run
        bool ok = true;
        float t_opt = 0;
        for (int r = 0; r < RUNS; r++) {
            CUDA_CHECK(cudaMemcpy(d_in, data[r].data(), (size_t)n * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            radix_sort(d_in, d_out, n);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            t_opt += time_ms(e0, e1);

            CUDA_CHECK(cudaMemcpy(res.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
            if (res != refs[r]) ok = false;
        }
        t_opt /= RUNS;

        printf("n = %9d  avg = %9.3f ms  (%s)\n", n, t_opt, ok ? "OK" : "FAIL");

        cudaFree(d_in); cudaFree(d_out);
    }
    return 0;
}
