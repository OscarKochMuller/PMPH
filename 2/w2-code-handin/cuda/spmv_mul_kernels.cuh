#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    int gid = blockIdx.x* blockDim.x +threadIdx.x;
    if (gid <tot_size) flags_d[gid] = 0;
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    int gid = blockIdx.x *blockDim.x + threadIdx.x;
    if (gid < mat_rows) { int start =(gid == 0) ? 0 : mat_shp_sc_d[gid - 1];
        flags_d[start ]=1;
    }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    int gid = blockIdx.x *blockDim.x + threadIdx.x;
    if (gid < tot_size) tmp_pairs[gid] = mat_vals[gid] *vct[mat_inds[gid]];
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid<mat_rows) res_vct_d[gid] =tmp_scan[mat_shp_sc_d[gid] - 1];
}

#endif // SPMV_MUL_KERNELS
