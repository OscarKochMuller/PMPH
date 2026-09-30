CXX        = nvcc -O3

SRC_GPU = main-gpu.cu
SRC_CPU = main-omp.c
HELPERS = ../helper.h parallelPlan.h goldenSeq.h
KERNELS = kernels.cu.h
EXEC_GPU = coalesce-gpu
EXEC_CPU = coalesce-cpu

default: compile_cpu run_cpu run_cpu_seq compile_gpu run_gpu

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile_cpu: $(EXEC_CPU)
compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU)

$(EXEC_CPU): $(SRC_CPU) $(HELPERS)
	g++ -O3 -fopenmp -o $(EXEC_CPU) $(SRC_CPU)

run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 1048576

run_cpu: $(EXEC_CPU)
	./$(EXEC_CPU) 1048576

run_cpu_seq: $(EXEC_CPU)
	OMP_NUM_THREADS=1 ./$(EXEC_CPU) 1048576

clean:
	rm -f $(EXEC_GPU) $(EXEC_CPU)

