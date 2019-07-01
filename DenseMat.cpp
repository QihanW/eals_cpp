#include <vector>

#include "DenseMat.h"
#include "DenseVec.h"
#include <random>

using namespace std;

DenseMat::DenseMat() {
	this->numColumns = 0;
	this->numRows = 0;;
}

DenseMat::DenseMat(int numRows, int numColumns) {
	this->numColumns = numColumns;
	this->numRows = numRows;
	this->matrix.resize(numRows);
	for (int i=0; i<numRows; i++){
    this->matrix[i].resize(numColumns);
  }
}

DenseMat::DenseMat(vector<vector<double>> mat) {;
	this->numColumns = mat[0].size();
	this->numRows = mat.size();
	this->matrix = mat;
}
/*
DenseMat::DenseMat(DenseMat mat){
  this->numRows = mat.numRows;
  this->numColumns = mat.numColumns;
	for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numColumns; j++) {
      this->matrix.push_back(mat.matrix[i][j]);
    }
  }
}*/

DenseMat DenseMat::clone() {
	DenseMat mat(this->matrix);
	return mat;
}

void DenseMat::init(double mean, double sigma) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, sigma);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numColumns; j++) {
			this->matrix[i][j] = distribution(generator);
		}
	}
}

void DenseMat::init(double range) {
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, range);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numColumns; j++) {
			this->matrix[i][j] = distribution(generator);
		}
	}
}

void DenseMat::init() {
	init(1.0);
}

void DenseMat::set(int row, int column, double val) {
	this->matrix[row][column] = val;
}

double DenseMat::get(int row, int column) {
	return this->matrix[row][column];
}

double DenseMat::squaredSum() {
	double res = 0;
  for (int i = 0; i < this-> numRows; i++)
    for (int j = 0; j < this->numColumns; j++)
      res += matrix[i][j] * matrix[i][j];
  return res;
}

DenseMat DenseMat::transpose() {
	DenseMat mat(this->numColumns, this->numRows);

		for (int i = 0; i < mat.numRows; i++)
			for (int j = 0; j < mat.numColumns; j++)
				mat.set(i, j, this->matrix[j][i]);

		return mat;
}

DenseVec DenseMat::row(int rowId) {

	DenseVec vec(this->matrix[rowId]);
	return vec;
}

DenseVec DenseMat::row_fal(int rowId){
  DenseVec tmp(this->matrix[rowId],false);
  return tmp;
}

DenseVec DenseMat::col(int colId) {
	DenseVec vec(this->numRows);
  for (int j = 0; j < this->numRows; j++)
	  vec.set(j, this->matrix[j][colId]);
	return vec;
}

DenseMat DenseMat::mult(DenseMat mat){
  DenseMat res(this->numRows, mat.numColumns);
		for (int i = 0; i < res.numRows; i++) {
			for (int j = 0; j < res.numColumns; j++) {

				double product = 0;
				for (int k = 0; k < this->numColumns; k++)
					product += matrix[i][k] * mat.matrix[k][j];

				res.set(i, j, product);
			}
		}

		return res;
}

DenseVec DenseMat::mult(DenseVec vec){
  DenseVec res(this->numRows);
  for (int i = 0; i < this->numRows; i++)
			res.set(i, row(i).inner(vec));

	return res;
}





