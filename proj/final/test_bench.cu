
#include <cub/cub.cuh>
#include "radix_sort.cuh"
static float time_ms(cudaEvent_t a,cudaEvent_t b) {float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

// writes Futhark data
static void write_futhark_u32(const char* path, const std::vector<uint32_t>& v)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(EXIT_FAILURE); }
    const unsigned char hdr[] = {'b', 2, 1, ' ', 'u', '3', '2'};
    const uint64_t len = v.size();
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(&len, sizeof(len), 1, f);
    fwrite(v.data(), sizeof(uint32_t), v.size(), f);
    fclose(f);
}

int main()
{
    
    const int sizes[]= {1000, 10000, 100000, 1000000, 10000000, 100000000};
    const int RUNS = 10;
    const float T_CRIT_95 = 2.262f;

    std::mt19937 rng(12345);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int n : sizes) {
        // RUNS independent datasets, shared by algorithms
        std::vector<std::vector<uint32_t>> data(RUNS), refs(RUNS);
        for (int r = 0; r < RUNS; r++) {
            data[r].resize(n);
            for (auto& x : data[r]) x = rng();
            refs[r] = data[r];
            std::sort(refs[r].begin(), refs[r].end());

            // save dataset for Futhark
            char path[64];
            snprintf(path, sizeof(path), "data_n%d_r%d.in", n, r);
            write_futhark_u32(path, data[r]);
        }

        uint32_t *d_in, *d_out, *d_cub_in, *d_cub_out;
        CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * 4));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * 4));
        CUDA_CHECK(cudaMalloc(&d_cub_in,(size_t)n * 4));
        CUDA_CHECK(cudaMalloc(&d_cub_out, (size_t)n * 4));

        std::vector<uint32_t> res(n);

        // CUB setup + warm-up
        void* d_tmp = nullptr; size_t tmp_bytes = 0;
        CUDA_CHECK(cudaMemcpy(d_cub_in, data[0].data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_cub_in, d_cub_out, n));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_cub_in, d_cub_out, n));
        CUDA_CHECK(cudaDeviceSynchronize());

        // runs of CUB
        bool cub_ok = true;
        std::vector<float> cub_times(RUNS);
        for (int r = 0; r < RUNS;r++) {
            CUDA_CHECK(cudaMemcpy(d_cub_in, data[r].data(), (size_t)n *4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(e0));
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_cub_in, d_cub_out, n));
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            cub_times[r] = time_ms(e0,e1);

            CUDA_CHECK(cudaMemcpy(res.data(), d_cub_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
            if (res != refs[r]) cub_ok = false;
        }

        // warm-up of our sort 
        CUDA_CHECK(cudaMemcpy(d_in, data[0].data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        radix_sort(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // timed runs of our sort
        bool ok = true;
        std::vector<float> opt_times(RUNS);
        for (int r = 0; r < RUNS; r++) {
            CUDA_CHECK(cudaMemcpy(d_in, data[r].data(), (size_t)n * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            radix_sort(d_in, d_out, n);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            opt_times[r] = time_ms(e0, e1);

            CUDA_CHECK(cudaMemcpy(res.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
            if (res != refs[r]) ok = false;
        }

        // print 
        printf("\n=== n = %d  (CUB=%s, opt=%s) ===\n", n, cub_ok ? "OK" : "FAIL", ok ? "OK" : "FAIL");
        printf("%6s %12s %12s %10s\n", "run", "CUB [ms]", "opt [ms]", "opt/CUB");
        float t_opt = 0, t_cub = 0;
        for (int r = 0; r < RUNS; r++) {
            printf("%6d %12.3f %12.3f %10.2f\n", r, cub_times[r], opt_times[r], opt_times[r] / cub_times[r]);
            t_cub += cub_times[r];
            t_opt += opt_times[r];
        }
        t_cub /= RUNS;
        t_opt /= RUNS;
        printf("%6s %12.3f %12.3f %10.2f\n", "avg", t_cub, t_opt, t_opt / t_cub);

        // 95% CI
        float var_cub = 0, var_opt = 0;
        for (int r = 0; r < RUNS; r++) {
            var_cub += (cub_times[r] - t_cub) * (cub_times[r] - t_cub);
            var_opt += (opt_times[r] - t_opt) * (opt_times[r] - t_opt);
        }
        float ci_cub = T_CRIT_95 * sqrtf(var_cub / (RUNS - 1)) / sqrtf((float)RUNS);
        float ci_opt = T_CRIT_95 * sqrtf(var_opt / (RUNS - 1)) / sqrtf((float)RUNS);
        printf("%6s [%.3f, %.3f] [%.3f, %.3f]\n", "95%CI", t_cub - ci_cub, t_cub + ci_cub, t_opt - ci_opt, t_opt + ci_opt);

        cudaFree(d_in); cudaFree(d_out);
        cudaFree(d_cub_in); cudaFree(d_cub_out); cudaFree(d_tmp);
    }
    return 0;}
