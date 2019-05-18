#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "DenseMat.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

DenseMat::DenseMat() {
	this->numColumns = 0;
	this->numRows = 0;
	this->metric.resize(0,0);
}

DenseMat::DenseMat(int numRows, int numColumns) {
	this->numColumns = numColumns;
	this->numRows = numRows;
	this->metric.resize(numRows, numColumns);
}
DenseMat::DenseMat(MatrixXd mat) {
	this->numColumns = mat.rows();
	this->numRows = mat.cols();
	this->metric = mat;
}
DenseMat DenseMat::clone() {
	MatrixXd mat = this->metric;
	return mat;
}
void DenseMat::init(double mean, double sigma) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, sigma);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numColumns; j++) {
			this->metric(i,j) = distribution(generator);
		}
	}
}
void DenseMat::init(double range) {
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, range);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numColumns; j++) {
			this->metric(i, j) = distribution(generator);
		}
	}
}
void DenseMat::init() {
	init(1.0);
}
void DenseMat::set(int row, int column, double val) {
	this->metric(row, column) = val;
}
double DenseMat::get(int row, int column) {
	return this->metric(row, column);
}
MatrixXd DenseMat::getData() {
	return this->metric;
}
double DenseMat::squaredNorm() {
	return this->metric.squaredNorm();
}
MatrixXd DenseMat::transpose() {
	return this->metric.transpose();
}
VectorXd DenseMat::row(int rowId) {
	return this->metric.row(rowId);
}
VectorXd DenseMat::col(int colId) {
	return this->metric.col(colId);
}