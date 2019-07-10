#ifndef DENSE_MAT_H
#define DENSE_MAT_H

#include <random>
#include "DenseVec.h"
#include <vector>

using namespace std;

class DenseMat {
public:
	//float[][] metric;
	float ** matrix;
	DenseMat();
	DenseMat(int numRows, int numColumns);
	DenseMat(float ** mat);
//	DenseMat(DenseMat mat);
	DenseMat clone();
	void init(float mean, float sigma);
	void init(float range);
	void init();
	void set(int row, int column, float val);
	float get(int row, int column);
	float squaredSum();
	DenseMat transpose();
	DenseVec row(int rowId);
	DenseVec row_fal(int rowId);
	DenseVec col(int colId);
	DenseMat mult(DenseMat mat);
	DenseVec mult(DenseVec vec);
	int numRows, numColumns;
	
};


#endif // !DENSE_MAT_H

