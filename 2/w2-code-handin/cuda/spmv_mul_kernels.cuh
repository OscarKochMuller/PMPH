#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    // TODO: fill in your implementation here ...
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    // TODO: fill in your implementation here ...
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // TODO: fill in your implementation here ...
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    // TODO: fill in your implementation here ...
}

#endif // SPMV_MUL_KERNELS
