Run the code:
to run the code write in terminal 'make' to run all.
to clean use 'make clean'

decpition of files:

    code files:

        partial_optimisation.cu: code for partial tests 

        radix_sort.cuh: implementation of radix sort 

        sort.fut: use radix sort from githubrepo 

        futhark.pkg: githubrepo link

    test files: 

        test_bench.cu: test for running the radix sort with different sizes agrinst CUB and futhark. run with: make run-bench.

        bench_futhark.py: script for running arrays with futhark code: run with: run-futhark

        test_data.cu: test differet array patterns and sizes. run with: make run-data.

        test_edge.cu: test edge cases of implementation. run with: make edge.

        test_sweep.cu: test for different values of B and Q.  run with: make sweep.
