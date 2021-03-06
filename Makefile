MKLROOT = /opt/intel/mkl
MKLLINK = -L$(MKLROOT)/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64  -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl
CXXFLAGS = -DMKL_ILP64 -I$(MKLROOT)/include -I/home/qwang/eals/eigen/ -g -O3 -qopenmp

OBJECT =  DenseMat.o DenseVec.o SparseMat.o SparseVec.o MF_fastALS.o

eals: main.o $(OBJECT)
	icpc -o eals main.o $(OBJECT) $(MKLLINK) 

main.o: MF_fastALS.h Rating.h SparseMat.h
	icpc -c $(CXXFLAGS) main.cpp
DenseMat.o: DenseMat.h
	icpc -c $(CXXFLAGS) DenseMat.cpp
DenseVec.o: DenseVec.h
	icpc -c $(CXXFLAGS) DenseVec.cpp
SparseMat.o: SparseMat.h
	icpc -c $(CXXFLAGS) SparseMat.cpp
SparseVec.o: SparseVec.h
	icpc -c $(CXXFLAGS) SparseVec.cpp
MF_fastALS.o: MF_fastALS.h DenseVec.h DenseMat.h SparseVec.h SparseMat.h Rating.h
	icpc -c $(CXXFLAGS) MF_fastALS.cpp




.PHONY : clean
clean:
	rm -f *.o



