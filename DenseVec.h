#ifndef DENSE_VEC_H
#define DENSE_VEC_H

#include <random>
#include <vector>
using namespace std;

//typedef Matrix<double, Dynamic, 1> VectorXd;

class DenseVec {
public: 
	DenseVec(int size);
	DenseVec(double* data);
	//DenseVec(vector<double> data);
	DenseVec(double* data, bool judge);
	//DenseVec(DenseVec vec);
	DenseVec clone();
	void init(double mean, double sigma);
	void init();
	void set(int idx, double val);
	double get(int idx);
	int getSize();
	double inner(DenseVec vec);
  ~DenseVec();

	double* vect;
	int size;
};


#endif // !DENSE_VEC_H

