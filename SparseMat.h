#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <vector>
#include <map>
#include <random>
#include "SparseVec.h"

using namespace std;
/*
typedef SparseVector<double> SpVec;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<double, RowMajor> SpMat_R;
*/
class SparseMat {
public:
	SparseMat();
	SparseMat(int r, int c);
	//SparseMat(SparseMat newMat);

	void setValue(int i, int j, double value);
	//void setValueC(int i, int j, double value);
	//void setValueR(int i, int j, double value);

	//double getValueC(int i, int j);
	//double getValueR(int i, int j);

	//void setMatC(SpMat matc);
	void setSize(int m, int n);

	SparseVec getRowRef(int index);
	
	SparseVec getColRef(int index);
	
  double getValue(int r, int c);
  int itemCount();

	int n_r;
	int n_c;
	vector<SparseVec> rows;
	vector<SparseVec> cols;
};

#endif // !SPARSE_MAT_H

