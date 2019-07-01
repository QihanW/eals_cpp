MKLROOT = /opt/intel/mkl
MKLLINK = -L$(MKLROOT)/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64  -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl
CXXFLAGS = -DMKL_ILP64 -I$(MKLROOT)/include -I/home/qwang/eals/eigen/ -g  

OBJECT =  DenseMat.o DenseVec.o SparseMat.o SparseVec.o MF_fastALS.o

eals: main.o $(OBJECT)
	g++ -o eals main.o $(OBJECT) $(MKLLINK) 

main.o: MF_fastALS.h Rating.h SparseMat.h
	g++ -c $(CXXFLAGS) main.cpp
DenseMat.o: DenseMat.h
	g++ -c $(CXXFLAGS) DenseMat.cpp
DenseVec.o: DenseVec.h
	g++ -c $(CXXFLAGS) DenseVec.cpp
SparseMat.o: SparseMat.h
	g++ -c $(CXXFLAGS) SparseMat.cpp
SparseVec.o: SparseVec.h
	g++ -c $(CXXFLAGS) SparseVec.cpp
MF_fastALS.o: MF_fastALS.h DenseVec.h DenseMat.h SparseVec.h SparseMat.h Rating.h
	g++ -c $(CXXFLAGS) MF_fastALS.cpp




.PHONY : clean
clean:
	rm -f *.o



