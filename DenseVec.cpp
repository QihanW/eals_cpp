
#include "DenseVec.h"
#include <random>

using namespace std;

DenseVec::DenseVec(int size) {
	//this->vect.resize(size);
	this->size = size;
	this->vect = new float[size];
}

DenseVec::DenseVec(float* data) {
	//this->vector = data;
	this->size = sizeof(data)/sizeof(float);
  this->vect = new float[this->size];
	for (int i=0; i<this->size; i++){
    this->vect[i] = data[i];
  }
}
/*
DenseVec::DenseVec(vector<float> data){
  this->size = data.size();
  //this->vect = data;
	for (int i=0; i<this->size; i++){
    this->vect.push_back(data[i]);
  }
}
*/
DenseVec::DenseVec(float* data, bool judge){
  if(judge){
    this->size = sizeof(data)/sizeof(float);
    this->vect = new float[this->size];
      //this->vect = data;
    for (int i=0; i<this->size; i++){
        this->vect[i] = data[i];
    }
  }
  else{
    this->size = sizeof(data)/sizeof(float);
    this->vect = data;
  }
}
/*
DenseVec::DenseVec(DenseVec vec){
  this->size = vec.size; 
	for (int i=0; i<this->size; i++){
    this->vect.push_back(vec.vect[i]);
  }
}*/

int DenseVec::getSize(){
  return (size);
}

DenseVec DenseVec::clone() {
	DenseVec vec(this->vect);
	return vec;
}

void DenseVec::init(float mean, float sigma) {
	std::default_random_engine e; 
	std::normal_distribution<float> n(mean, sigma);
	for (int i = 0; i < this->size; i++) {
		this->vect[i] = n(e);
	}
}

void DenseVec::init() {
  
	std::default_random_engine e; 
	std::normal_distribution<float> n(0, 1);
	for (int i = 0; i < this->size; i++) {
	  this->vect[i] = n(e);
	}
}

float DenseVec::get(int idx) {
	return this->vect[idx];
}

void DenseVec::set(int idx, float val) {
	this->vect[idx] = val;
}

float DenseVec::inner(DenseVec vec){
  float result = 0;
  int size = vec.size;
  float *v1 = this->vect;
  float *v2 = vec.vect;
	for (int i = 0; i < size; i++)
			result += (*(v1+i)) * (*(v2+i));

		return result;
}

DenseVec::~DenseVec(){
//  delete [] this->vect;
}


