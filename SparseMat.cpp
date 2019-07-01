#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include "SparseMat.h"

using namespace Eigen;

typedef SparseVector<double> SpVec;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<double, RowMajor> SpMat_R;


SparseMat::SparseMat() {
	this->n_c = 0;
	this->n_r = 0;
}

SparseMat::SparseMat(int r, int c) {
	this->n_c = c;
	this->n_r = r;
}

SparseMat::SparseMat(SpMat matc, SpMat matr) {
	this->mat_c = matc;
	this->mat_r = matr;
}

void SparseMat::setMatC(SpMat matc) {
	this->mat_c = matc;
}


void SparseMat::setValue(int i, int j, double value) {
	this->mat_c.insert(i,j) = value;
	this->mat_r.insert(i,j) = value;
}

void SparseMat::setValueC(int i, int j, double value) {
	this->mat_c.insert(i, j) = value;
}

void SparseMat::setValueR(int i, int j, double value) {
	this->mat_r.insert(i, j) = value;
}

double SparseMat::getValueC(int i, int j) {
	return this->mat_c.coeff(i, j);
}

double SparseMat::getValueR(int i, int j) {
	return this->mat_r.coeff(i, j);
}

void SparseMat::setSize(int m, int n) {
	this->n_c = n;
	this->n_r = m;
}

double SparseMat::getRowOutIndex(int index) {
	return this->mat_r.outerIndexPtr()[index];
}

double SparseMat::getRowInIndex(int index) {
	return this->mat_r.innerIndexPtr()[index];
}

double SparseMat::getColOutIndex(int index) {
	return this->mat_c.outerIndexPtr()[index];
}

double SparseMat::getColInIndex(int index) {
	return this->mat_c.innerIndexPtr()[index];
}
