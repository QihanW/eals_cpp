icpc -c -DMKL_ILP64 -I/opt/intel/mkl/include -I/home/qwang/eals/eigen/ -g -O3 reorder.cpp -o -L/opt/intel/mkl/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64  -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl -L/usr/local/cuda/lib64 -lcudart reorder