#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

#include <random>
#include <map>
#include <vector>

using namespace std;

//typedef SparseVector<double> SpVec;

class SparseVec {
public:
	SparseVec();
	SparseVec(int num);
	//SparseVec(SpVec sv);
	void setValue(int i, double value);
	void setVector(SparseVec newVector);
	double getValue(int i);
	void setLength(int num);
	int itemCount();
	SparseVec getVector();
	vector<int> indexList();
	int n;
	map<int, double> spv;
};
#endif // !SPARSE_VECTO#R_H
