NVCC      ?= nvcc
NVCCFLAGS ?= -O3
PYTHON    ?= python3

SORTS_LIB = lib/github.com/diku-dk/sorts/radix_sort.fut
CUDA_OUT  = cuda_results.txt
FUT_OUT   = futhark_results.txt

.PHONY: all build run run-cuda run-futhark clean clean-data

# plain `make` builds everything and runs both benchmarks
all: run

build: test9 sort

# --- build ---------------------------------------------------------------
test9: test9.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SORTS_LIB):
	[ -f futhark.pkg ] || futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync

sort: sort.fut $(SORTS_LIB)
	futhark cuda sort.fut

# --- run (the Futhark benchmark needs the data files written by test9) ---
run: run-cuda run-futhark

run-cuda: $(CUDA_OUT)

$(CUDA_OUT): test9
	./test9 | tee $@

run-futhark: $(FUT_OUT)

$(FUT_OUT): sort bench_futhark.py $(CUDA_OUT)
	$(PYTHON) bench_futhark.py | tee $@

# --- clean ---------------------------------------------------------------
clean:
	rm -f test9 sort sort.c $(CUDA_OUT) $(FUT_OUT)

clean-data:
	rm -f data_n*_r*.in
