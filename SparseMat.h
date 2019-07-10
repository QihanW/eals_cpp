#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <vector>
#include <map>
#include <random>
#include "SparseVec.h"

using namespace std;
/*
typedef SparseVector<float> SpVec;
typedef SparseMatrix<float> SpMat;
typedef SparseMatrix<float, RowMajor> SpMat_R;
*/
class SparseMat {
public:
	SparseMat();
	SparseMat(int r, int c);
	//SparseMat(SparseMat newMat);

	void setValue(int i, int j, float value);
	//void setValueC(int i, int j, float value);
	//void setValueR(int i, int j, float value);

	//float getValueC(int i, int j);
	//float getValueR(int i, int j);

	//void setMatC(SpMat matc);
	void setSize(int m, int n);

	SparseVec getRowRef(int index);
	
	SparseVec getColRef(int index);
	
  float getValue(int r, int c);
  int itemCount();

	int n_r;
	int n_c;
	SparseVec *rows;
	SparseVec *cols;
};

#endif // !SPARSE_MAT_H

