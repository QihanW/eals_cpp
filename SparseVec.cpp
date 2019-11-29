#include <vector>
#include <map>
#include <random>
#include "SparseVec.h"

using namespace std;
//typedef SparseVector<float> SpVec;

SparseVec::SparseVec() {
	this->n = 0;
	this->current = 0;
	this->spv_in = new int[0];
	this->spv_do = new float[0];
}

SparseVec::SparseVec(int num) {
	this->n = num;
	this->current = 0;
	this->spv_in = new int[num];
	this->spv_do = new float[num];
}
/*
SparseVec::SparseVec(SpVec sv) {
	this->n = sv.size();
	this->vector = sv;
}*/

void SparseVec::setValue(int i, float value) {
  *(this->spv_in + this->current) = i;
  *(this->spv_do + this->current) = value;
  this->current++;
	//this->n++;
}

void SparseVec::setVector(SparseVec newVector) {
	this->n = newVector.n;
	this->spv_in = new int[this->n];
	this->spv_do = new float[this->n];
	for(int i=0; i<this->n; i++){
    this->spv_in[i] = newVector.spv_in[i];
    this->spv_do[i] = newVector.spv_do[i];
  }
}

float SparseVec::getValue(int i){
	for(int j=0; j<this->n; j++){
    if(this->spv_in[j] == i)
      return this->spv_do[j];
  }
  return 0;
}

void SparseVec::setLength(int num) {
	this->n = num;
	this->spv_in = new int[num];
	this->spv_do = new float[num];
}

int SparseVec::itemCount() {
	return this->n;
}

SparseVec SparseVec::getVector() {
  SparseVec tmp(this->n);
  for(int i=0; i<this->n; i++){
    tmp.setValue(this->spv_in[i],this->spv_do[i]);
  }
  return tmp;
}

vector<int> SparseVec::indexList(){
   vector<int> tmp;
   for(int i=0; i<this->n; i++){
    tmp.insert(tmp.end(), this->spv_in[i]);
   } 
   return tmp;
}
