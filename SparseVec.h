#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

#include <random>
#include <map>
#include <vector>

using namespace std;

//typedef SparseVector<float> SpVec;

class SparseVec {
public:
	SparseVec();
	SparseVec(int num);
	//SparseVec(SpVec sv);
	void setValue(int i, float value);
	void setVector(SparseVec newVector);
	float getValue(int i);
	void setLength(int num);
	int itemCount();
	SparseVec getVector();
  vector<int> indexList();
	int n;
	int current;
	int *spv_in;
	float *spv_do;
};
#endif // !SPARSE_VECTO#R_H
