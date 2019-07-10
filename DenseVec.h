#ifndef DENSE_VEC_H
#define DENSE_VEC_H

#include <random>
#include <vector>
using namespace std;

//typedef Matrix<double, Dynamic, 1> VectorXd;

class DenseVec {
public: 
	DenseVec(int size);
	DenseVec(float* data);
	//DenseVec(vector<double> data);
	DenseVec(float* data, bool judge);
	//DenseVec(DenseVec vec);
	DenseVec clone();
	void init(float mean, float sigma);
	void init();
	void set(int idx, float val);
	float get(int idx);
	int getSize();
	float inner(DenseVec vec);
  ~DenseVec();

	float* vect;
	int size;
};


#endif // !DENSE_VEC_H

