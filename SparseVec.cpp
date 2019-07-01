#include <vector>
#include <map>
#include <random>
#include "SparseVec.h"

using namespace std;
//typedef SparseVector<double> SpVec;

SparseVec::SparseVec() {
	this->n = 0;;
}

SparseVec::SparseVec(int num) {
	this->n = num;
}
/*
SparseVec::SparseVec(SpVec sv) {
	this->n = sv.size();
	this->vector = sv;
}*/

void SparseVec::setValue(int i, double value) {
	this->spv.insert(pair<int, double>(i, value));
}

void SparseVec::setVector(SparseVec newVector) {
	this->n = newVector.n;
	map<int, double>::iterator iter;  
  for(iter = newVector.spv.begin(); iter != newVector.spv.end(); iter++){
    this->spv.insert(pair<int, double>(iter->first, iter->second));
  }
}

double SparseVec::getValue(int i){
	map<int, double>::iterator iter;
	iter = this->spv.find(i);
	return iter->second;
}

void SparseVec::setLength(int num) {
	this->n = num;
}

int SparseVec::itemCount() {
	return this->spv.size();
}

SparseVec SparseVec::getVector() {
  SparseVec tmp(this->n);
	map<int, double>::iterator iter;  
  for(iter = this->spv.begin(); iter != this->spv.end(); iter++){
    tmp.setValue(iter->first, iter->second);
  }
  return tmp;
}

vector<int> SparseVec::indexList(){
  vector<int> tmp;
	map<int, double>::iterator iter;  
  for(iter = this->spv.begin(); iter != this->spv.end(); iter++){
    tmp.insert(tmp.end(), iter->first);      
  }
  return tmp;
}
