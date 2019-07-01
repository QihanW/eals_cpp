
#include "DenseVec.h"
#include <random>

using namespace std;

DenseVec::DenseVec(int size) {
	this->vect.resize(size);
	this->size = size;
}

DenseVec::DenseVec(double *data) {
	//this->vector = data;
	this->size = sizeof(data)/sizeof(double);
	for (int i=0; i<this->size; i++){
    this->vect.push_back(data[i]);
  }
}

DenseVec::DenseVec(vector<double> data){
  this->size = data.size();
  //this->vect = data;
	for (int i=0; i<this->size; i++){
    this->vect.push_back(data[i]);
  }
}

DenseVec::DenseVec(vector<double> data, bool judge){
  if(judge){
    this->size = data.size();
      //this->vect = data;
    for (int i=0; i<this->size; i++){
        this->vect.push_back(data[i]);
      }
  }
  else{
    this->size = data.size();
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

void DenseVec::init(double mean, double sigma) {
	std::default_random_engine e; 
	std::normal_distribution<double> n(mean, sigma);
	for (int i = 0; i < this->size; i++) {
		this->vect[i] = n(e);
	}
}

void DenseVec::init() {
  
	std::default_random_engine e; 
	std::normal_distribution<double> n(0, 1);
	for (int i = 0; i < this->size; i++) {
	  this->vect[i] = n(e);
	}
}

double DenseVec::get(int idx) {
	return this->vect[idx];
}

void DenseVec::set(int idx, double val) {
	this->vect[idx] = val;
}

double DenseVec::inner(DenseVec vec){
  double result = 0;
  int size = vec.size;
	for (int i = 0; i < size; i++)
			result += this->vect[i] * vec.vect[i];

		return result;
}
