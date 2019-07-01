#ifndef DENSE_MAT_H
#define DENSE_MAT_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<double, Dynamic, 1> VectorXd;

class DenseMat {
public:
	MatrixXd metric;
	DenseMat();
	DenseMat(int numRows, int numColumns);
	DenseMat(MatrixXd mat);
	DenseMat clone();
	void init(double mean, double sigma);
	void init(double range);
	void init();
	void set(int row, int column, double val);
	double get(int row, int column);
	MatrixXd getData();
	double squaredNorm();
	MatrixXd transpose();
	VectorXd row(int rowId);
	VectorXd col(int colId);
protected:
	int numRows, numColumns;
	
};


#endif // !DENSE_MAT_H

