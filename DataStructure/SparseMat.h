#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef SparseVector<double> SpVec;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<double, RowMajor> SpMat_R;

class SparseMat {
public:
	SparseMat();
	SparseMat(int r, int c);
	SparseMat(SpMat matc, SpMat matr);

	void setValue(int i, int j, double value);
	void setValueC(int i, int j, double value);
	void setValueR(int i, int j, double value);

	double getValueC(int i, int j);
	double getValueR(int i, int j);

	void setMatC(SpMat matc);
	void setSize(int m, int n);

	double getRowOutIndex(int index);
	double getRowInIndex(int index);
	double getColOutIndex(int index);
	double getColInIndex(int index);

private:
	int n_r;
	int n_c;
	SpMat mat_c;
	SpMat_R mat_r;
};

#endif // !SPARSE_MAT_H

