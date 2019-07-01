#ifndef DENSE_MAT_H
#define DENSE_MAT_H

#include <random>
#include "DenseVec.h"
#include <vector>

using namespace std;

class DenseMat {
public:
	//double[][] metric;
	double ** matrix;
	DenseMat();
	DenseMat(int numRows, int numColumns);
	DenseMat(double ** mat);
//	DenseMat(DenseMat mat);
	DenseMat clone();
	void init(double mean, double sigma);
	void init(double range);
	void init();
	void set(int row, int column, double val);
	double get(int row, int column);
	double squaredSum();
	DenseMat transpose();
	DenseVec row(int rowId);
	DenseVec row_fal(int rowId);
	DenseVec col(int colId);
	DenseMat mult(DenseMat mat);
	DenseVec mult(DenseVec vec);
	int numRows, numColumns;
	
};


#endif // !DENSE_MAT_H

