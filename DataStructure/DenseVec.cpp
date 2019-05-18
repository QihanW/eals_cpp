#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "DenseVec.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> VectorXd;

DenseVec::DenseVec(int size) {
	this->vector.resize(size);
	this->size = size;
}
DenseVec::DenseVec(VectorXd data) {
	this->vector = data;
	this->size = data.size();
}
DenseVec DenseVec::clone() {
	VectorXd vec = this->vector;
	return vec;
}
void DenseVec::init(double mean, double sigma) {
	std::default_random_engine e; 
	std::normal_distribution<double> n(mean, sigma);
	for (int i = 0; i < size; i++) {
		this->vector[i] = n(e);
	}
}
void DenseVec::init() {
	this->vector = VectorXd::Random(size, 1);
}
double DenseVec::get(int idx) {
	return this->vector[idx];
}
VectorXd DenseVec::getData() {
	return this->vector;
}
void DenseVec::set(int idx, double val) {
	this->vector[idx] = val;
}
