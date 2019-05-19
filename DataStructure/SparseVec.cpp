#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include "SparseVec.h"

using namespace Eigen;
typedef SparseVector<double> SpVec;

SparseVec::SparseVec() {
	this->n = 0;
	this->vector = SpVec(0);
}

SparseVec::SparseVec(int num) {
	this->n = num;
	this->vector = SpVec(num);
}

SparseVec::SparseVec(SpVec sv) {
	this->n = sv.size();
	this->vector = sv;
}

void SparseVec::setValue(int i, double value) {
	this->vector.coeffRef(i) = value;
}

void SparseVec::setVector(SparseVec newVector) {
	this->n = newVector.getLength();
	this->vector = newVector.getVector();
}

double SparseVec::getValue(int i) {
	return this->vector.coeffRef(i);
}

void SparseVec::setLength(int num) {
	this->n = num;
}

int SparseVec::getLength() {
	return this->n;
}

SpVec SparseVec::getVector() {
	return this->vector;
}