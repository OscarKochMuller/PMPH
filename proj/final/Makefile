NVCC      ?= nvcc
NVCCFLAGS ?= -O3
PYTHON    ?= python3
ARCH         ?= native
SWEEP_BLOCKS ?= 128 256 512
SWEEP_QKEYS  ?= 16 22 28
SORTS_LIB = lib/github.com/diku-dk/sorts/radix_sort.fut
CUDA_OUT  = cuda_results.txt
FUT_OUT   = futhark_results.txt
PARTIAL_OUT = partial_results.txt
DATA_OUT = data_results.txt

.PHONY: all build run run-bench run-cuda run-partial run-futhark edge sweep-grid clean clean-data

all: run
build: test_bench sort partial_optimisation

# --- build ---------------------------------------------------------------
test_bench: test_bench.cu radix_sort.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 -arch=sm_80 -Xptxas -v $< -o $@

$(SORTS_LIB):
	[ -f futhark.pkg ] || futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync

sort: sort.fut $(SORTS_LIB) 
	futhark cuda sort.fut

partial_optimisation: partial_optimisation.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 $< -o $@


test_data: test_data.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# --- run ---------------------------------------------------------------

run: run-bench run-futhark edge sweep-grid run-partial run-data

run-partial: partial_optimisation
	./partial_optimisation | tee $(PARTIAL_OUT)

run-bench: test_bench
	./test_bench | tee $(CUDA_OUT)

run-data: test_data
	./test_data | tee $(CUDA_OUT)

run-futhark: sort bench_futhark.py run-bench
	$(PYTHON) bench_futhark.py | tee $(FUT_OUT)

	
# --- edge and sweep tests ------------------------------------------------
edge: test_edge.cu radix_sort.cuh
	@echo ">>> edge-case tests, B=256 Q=22"
	@if $(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DBLOCK=256 -DQKEYS=22 test_edge.cu -o edge_B256_Q22 2> build_edge_B256_Q22.log; then \
	  ./edge_B256_Q22 | tee edge_B256_Q22.txt; \
	else \
	  echo "    edge test build failed, see build_edge_B256_Q22.log"; \
	fi


sweep-grid: test_sweep.cu radix_sort.cuh
	@for b in $(SWEEP_BLOCKS); do \
	  for q in $(SWEEP_QKEYS); do \
	    tag=B$${b}_Q$${q}; \
	    if $(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DBLOCK=$$b -DQKEYS=$$q test_sweep.cu -o sort_$$tag 2> build_$$tag.log; then \
	      if ./sort_$$tag > out_$$tag.txt 2>&1; then \
	        echo "B=$$b Q=$$q  ok      -> out_$$tag.txt"; \
	      else \
	        echo "B=$$b Q=$$q  RUN FAILED (see out_$$tag.txt)"; \
	      fi; \
	    else \
	      echo "B=$$b Q=$$q  SKIPPED (does not compile, see build_$$tag.log)"; \
	    fi; \
	  done; \
	done



# --- clean ---------------------------------------------------------------
clean: clean-data
	rm -f test_bench partial_optimisation sort sort.c test_data
	rm -f $(CUDA_OUT) $(FUT_OUT) $(PARTIAL_OUT)
	rm -f edge_B*_Q* sort_B*_Q* build_*.log out_B*_Q*.txt
	rm -f data_n*_r*.in
