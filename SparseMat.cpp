#include <vector>
#include <map>
#include <random>
#include "SparseMat.h"
#include "SparseVec.h"

using namespace std;

/*
typedef SparseVector<double> SpVec;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<double, RowMajor> SpMat_R;
*/

SparseMat::SparseMat() {
	this->n_c = 0;
	this->n_r = 0;
	this->rows.resize(0);
	this->cols.resize(0);
}

SparseMat::SparseMat(int r, int c) {
	this->n_c = c;
	this->n_r = r;
	this->rows.resize(r);
	this->cols.resize(c);
}
/*
SparseMat::SparseMat(SparseMat newMat) {
	this->n_c = newMat->n_c;
	this->n_r = newMat->n_r;
  this->rows.resize(this->n_r);
  this->cols.resize(this->n_c);
  
  for(int i=0; i<n_r; i++){
    this->rows[i] =  newMat->rows[i].getVector();
  }
  
  for(int i=0; i<n_c; i++){
    this->cols[i] =  newMat->cols[i].getVector();
  }
}

void SparseMat::setMatC(SpMat matc) {
	this->mat_c = matc;
}
*/

void SparseMat::setValue(int i, int j, double value) {
	this->rows[i].setValue(j, value);
	this->cols[j].setValue(i, value);
}
/*
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
*/
void SparseMat::setSize(int m, int n) {
	this->n_r = n;
	this->n_c = m;
}

SparseVec SparseMat::getRowRef(int index) {
	return this->rows[index];
}
/*
double SparseMat::getRowInIndex(int index) {
	return this->mat_r.innerIndexPtr()[index];
}*/

SparseVec SparseMat::getColRef(int index) {
	return this->cols[index];
}
/*
double SparseMat::getColInIndex(int index) {
	return this->mat_c.innerIndexPtr()[index];
}*/

double SparseMat::getValue(int r, int c){
  return this->rows[r].getValue(c);
}

int SparseMat::itemCount(){
  int sum = 0;
		
		if (n_r > n_c) {
			for (int i = 0; i < n_r; i++) {
				sum += rows[i].itemCount();
			}
		}
		else {
			for (int j = 0; j < n_c; j++) {
				sum += cols[j].itemCount();
			}
		}
		
		return sum;
}



