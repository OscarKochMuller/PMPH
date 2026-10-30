#include "radix_sort.cuh"


static bool test_size(int n, std::mt19937& rng, const char** failed_pattern)
{
    static const char* names[] = {"random", "sorted", "reversed", "all-zero", "all-max", "few-distinct"};
    const int NPAT = 6;

    std::vector<uint32_t> d(n), ref(n), res(n);
    uint32_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * 4));

    bool ok = true;
    for (int p = 0; p < NPAT && ok; p++) {
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
        ref = d;
        std::sort(ref.begin(), ref.end());

        CUDA_CHECK(cudaMemcpy(d_in, d.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0xA5, (size_t)n * 4));   
        radix_sort(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));

        if (res != ref) { ok = false; *failed_pattern = names[p]; }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

static int run_edge_tests()
{
    std::vector<int> sz = {
        6, 12, 24, 48, 6000,                      
        1, 2,
        Q / 2,                                    
        Q - 1, Q, Q + 1,                          
        2 * Q, 2 * Q + 1,                         
        32 * Q, 32 * Q + 1,                       
        TILE / 2,                                 
        TILE - 1, TILE, TILE + 1,                 
        2 * TILE, 2 * TILE + 1,                   
        3 * TILE + 17                             
    };
    std::sort(sz.begin(), sz.end());
    sz.erase(std::unique(sz.begin(), sz.end()), sz.end());

    std::mt19937 rng(777);
    int failures = 0;

    printf("\n=== edge-case tests (B=%d, Q=%d, TILE=%d) ===\n", B, Q, TILE);
    printf("%9s %7s %16s %18s  %s\n", "n", "blocks", "keys in last blk", "threads w/ keys", "result");
    for (int n : sz) {
        if (n < 1) continue;
        const int blocks = (n + TILE - 1) / TILE;
        const int last   = n - (blocks - 1) * TILE;          
        const int thr    = (last + Q - 1) / Q;               
        const char* why = "";
        const bool ok = test_size(n, rng, &why);
        if (!ok) failures++;
        printf("%9d %7d %16d %18d  %s%s%s\n", n, blocks, last, thr,
               ok ? "OK" : "FAIL (", ok ? "" : why, ok ? "" : ")");
    }
    printf("edge-case tests: %s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures;
}

int main()
{
    printf("CONFIG: B=%d Q=%d LGH=%d passes=%d TILE=%d\n", B, Q, LGH, (32 + LGH - 1) / LGH, TILE);
    return run_edge_tests() ? 1 : 0;
}
