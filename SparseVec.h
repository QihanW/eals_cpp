#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef SparseVector<double> SpVec;

class SparseVec {
public:
	SparseVec();
	SparseVec(int num);
	SparseVec(SpVec sv);
	void setValue(int i, double value);
	void setVector(SparseVec newVector);
	double getValue(int i);
	void setLength(int num);
	int getLength();
	SpVec getVector();
private:
	int n;
	SpVec vector;
};
#endif // !SPARSE_VECTOR_H
