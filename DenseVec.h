#ifndef DENSE_VEC_H
#define DENSE_VEC_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> VectorXd;

class DenseVec {
public: 
	DenseVec(int size);
	DenseVec(VectorXd data);
	DenseVec clone();
	void init(double mean, double sigma);
	void init();
	void set(int idx, double val);
	double get(int idx);
	VectorXd getData();
protected:
	VectorXd vector;
	int size = 0;
};


#endif // !DENSE_VEC_H

