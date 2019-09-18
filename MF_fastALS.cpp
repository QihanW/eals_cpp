#include "omp.h"
#include "MF_fastALS.h"
#include <math.h>
#include <vector>
#include <float.h>
#include <random>
#include <chrono>
#include <string>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <functional>
#include <time.h>
#include <immintrin.h>
#include "DenseVec.h"
#include "DenseMat.h"
#include "SparseVec.h"
#include "SparseMat.h"
#include "Rating.h"

#define NUM_float 8

#define GETVALUE(f, k, val, SV) \
  val = SV.matrix[f][k];

#define SUUPDATE(f, k, val) \
  tmp = 0 - u_clone.matrix[u][f] * u_clone.matrix[u][k] + U.matrix[u][f] * U.matrix[u][k];\
  val += tmp;

#define SVUPDATE(f, k, val) \
  tmp = (0 - v_clone.matrix[u][f] * v_clone.matrix[u][k] + V.matrix[u][f] * V.matrix[u][k]) *  Wi[u];\
  val += tmp;

#define SETVALUE(f, k, val, SV) \
  SV.matrix[f][k] = val; \
  SV.matrix[k][f] = val;

MF_fastALS::MF_fastALS(SparseMat trainMatrix1, std::vector<Rating> testRatings1, int topK1, int threadNum1, int factors1, int maxIter1, float w01, float alpha1, float reg1, float init_mean1, float init_stdev1, bool showProgress1, bool showLoss1, int userCount1, int itemCount1){
	trainMatrix = trainMatrix1;
	testRatings = testRatings1;
	topK = topK1;
	factors = factors1;
	maxIter = maxIter1;
	w0 = w01;
	reg = reg1;
	alpha = alpha1;
	init_mean = init_mean1;
	init_stdev = init_stdev1;
	showloss = showLoss1;
	showprogress = showProgress1;
	itemCount = itemCount1;
	userCount = userCount1;

	// Set the Wi as a decay function w0 * pi ^ alpha
	float sum = 0, Z = 0;
	float *p = new float[itemCount];
	for (int i = 0; i < itemCount; i++) {
		p[i] = trainMatrix.getColRef(i).itemCount();
		sum += p[i];
	}

	// convert p[i] to probability 
	for (int i = 0; i < itemCount; i++) {
		p[i] /= sum;
		p[i] = pow(p[i], alpha);
		Z += p[i];
	}
	// assign weight
	Wi = new float[itemCount];
	for (int i = 0; i < itemCount; i++)
		Wi[i] = w0 * p[i] / Z;

	// By default, the weight for positive instance is uniformly 1
	W = SparseMat(userCount, itemCount);
	for (int u = 0; u < userCount; u++){                                          
    W.rows[u].setVector(trainMatrix.rows[u]);                                   
  }                                                                             
  for (int i = 0; i < itemCount; i++){                                          
    W.cols[i].setVector(trainMatrix.cols[i]);                                   
  } 

  //Init model parameters
	U = DenseMat(userCount, factors);
	V = DenseMat(itemCount, factors);

	U.init(init_mean, init_stdev);
	V.init(init_mean, init_stdev);
	initS();

}

void MF_fastALS::setTrain(SparseMat trainMatrix) {
	this->trainMatrix = trainMatrix;
	
	W = SparseMat(userCount, itemCount);
	for (int u = 0; u < userCount; u++){
    W.rows[u].setVector(trainMatrix.rows[u]);
  }
  for (int i = 0; i < itemCount; i++){
    W.cols[i].setVector(trainMatrix.cols[i]);
  }  
}

void MF_fastALS::setUV(DenseMat U, DenseMat V) {
	this->U = U.clone();
	this->V = V.clone();
	initS();
}

void MF_fastALS::buildModel() {
	omp_set_num_threads(1);
	float loss_pre = FLT_MAX;

	for (int iter = 0; iter < maxIter; iter++) {
		int user_list = 0, item_list = 0;
	  
		DenseMat u_clone(userCount, factors);
		#pragma omp parallel for
		for (int i = 0; i<userCount; i++){
      for (int j = 0; j<factors; j++){
        u_clone.matrix[i][j] = U.matrix[i][j];
      }
    }
 
		double start = omp_get_wtime();
		#pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi)
		for (int u = 0; u < userCount; u++) {
			update_user_thread(u);  
		}
    
    double end = omp_get_wtime();
    float tmp, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    std::cout<<"First part time: "<<end-start<<std::endl;
    for (int f = 0; f < factors; f+=4) {                                                                                           
	    for (int k = 0; k <= f; k+=4) {
	      GETVALUE(f, k, val1, SU);
        GETVALUE(f, k+1, val2, SU);
        GETVALUE(f, k+2, val3, SU);
        GETVALUE(f, k+3, val4, SU);
        GETVALUE(f+1, k, val5, SU);
        GETVALUE(f+1, k+1, val6, SU);
        GETVALUE(f+1, k+2, val7, SU);
        GETVALUE(f+1, k+3, val8, SU);
        GETVALUE(f+2, k, val9, SU);
        GETVALUE(f+2, k+1, val10, SU);
        GETVALUE(f+2, k+2, val11, SU);
        GETVALUE(f+2, k+3, val12, SU);
        GETVALUE(f+3, k, val13, SU);
        GETVALUE(f+3, k+1, val14, SU);
        GETVALUE(f+3, k+2, val15, SU);
        GETVALUE(f+3, k+3, val16, SU);
	      #pragma omp parallel for schedule(dynamic, 64) reduction(+: val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16) 
	      for (int u = 0; u < userCount; u++){
		      SUUPDATE(f, k, val1);
          SUUPDATE(f, k+1, val2);
          SUUPDATE(f, k+2, val3);
          SUUPDATE(f, k+3, val4);
          SUUPDATE(f+1, k, val5);
          SUUPDATE(f+1, k+1, val6);
          SUUPDATE(f+1, k+2, val7);
          SUUPDATE(f+1, k+3, val8);
          SUUPDATE(f+2, k, val9);
          SUUPDATE(f+2, k+1, val10);
          SUUPDATE(f+2, k+2, val11);
          SUUPDATE(f+2, k+3, val12);
          SUUPDATE(f+3, k, val13);
          SUUPDATE(f+3, k+1, val14);
          SUUPDATE(f+3, k+2, val15);
          SUUPDATE(f+3, k+3, val16);
        }
		    SETVALUE(f, k, val1, SU);
        SETVALUE(f, k+1, val2, SU);
        SETVALUE(f, k+2, val3, SU);
        SETVALUE(f, k+3, val4, SU);
        SETVALUE(f+1, k, val5, SU);
        SETVALUE(f+1, k+1, val6, SU);
        SETVALUE(f+1, k+2, val7, SU);
        SETVALUE(f+1, k+3, val8, SU);
        SETVALUE(f+2, k, val9, SU);
        SETVALUE(f+2, k+1, val10, SU);
        SETVALUE(f+2, k+2, val11, SU);
        SETVALUE(f+2, k+3, val12, SU);
        SETVALUE(f+3, k, val13, SU);
        SETVALUE(f+3, k+1, val14, SU);
        SETVALUE(f+3, k+2, val15, SU);
        SETVALUE(f+3, k+3, val16, SU);                                                                                                                                
      }     
    }
    std::cout<<"The second part: "<<omp_get_wtime() - end<<std::endl; 
    double time_user_update = omp_get_wtime() - start;
	  std::cout << "Time of user_update: " <<time_user_update<< std::endl;
    
	  DenseMat v_clone(itemCount, factors);
    for (int i = 0; i<itemCount; i++){
	    for (int j = 0; j<factors; j++){
		    v_clone.matrix[i][j] = V.matrix[i][j];
	    }
    }
	  
	  start = omp_get_wtime();
	  #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SU, V, Wi)
	  for (int i = 0; i < itemCount; i++) {
      update_item_thread(i);
	  }

    for (int f = 0; f < factors; f+=4) {
	    for (int k = 0; k <= f; k+=4) {
		    GETVALUE(f, k, val1, SV);
        GETVALUE(f, k+1, val2, SV);
        GETVALUE(f, k+2, val3, SV);
        GETVALUE(f, k+3, val4, SV);
        GETVALUE(f+1, k, val5, SV);
        GETVALUE(f+1, k+1, val6, SV);
        GETVALUE(f+1, k+2, val7, SV);
        GETVALUE(f+1, k+3, val8, SV);
        GETVALUE(f+2, k, val9, SV);
        GETVALUE(f+2, k+1, val10, SV);
        GETVALUE(f+2, k+2, val11, SV);
        GETVALUE(f+2, k+3, val12, SV);
        GETVALUE(f+3, k, val13, SV);
        GETVALUE(f+3, k+1, val14, SV);
        GETVALUE(f+3, k+2, val15, SV);
        GETVALUE(f+3, k+3, val16, SV);
 		    #pragma omp parallel for schedule(dynamic, 64) reduction(+:val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16)
  		  for (int u = 0; u < itemCount; u++){
 			    SVUPDATE(f, k, val1);
          SVUPDATE(f, k+1, val2);
          SVUPDATE(f, k+2, val3);
          SVUPDATE(f, k+3, val4);
          SVUPDATE(f+1, k, val5);
          SVUPDATE(f+1, k+1, val6);
          SVUPDATE(f+1, k+2, val7);
          SVUPDATE(f+1, k+3, val8);
          SVUPDATE(f+2, k, val9);
          SVUPDATE(f+2, k+1, val10);
          SVUPDATE(f+2, k+2, val11);
          SVUPDATE(f+2, k+3, val12);
          SVUPDATE(f+3, k, val13);
          SVUPDATE(f+3, k+1, val14);
          SVUPDATE(f+3, k+2, val15);
          SVUPDATE(f+3, k+3, val16);
		    }
        SETVALUE(f, k, val1, SV);
        SETVALUE(f, k+1, val2, SV);
        SETVALUE(f, k+2, val3, SV);
        SETVALUE(f, k+3, val4, SV);
        SETVALUE(f+1, k, val5, SV);
        SETVALUE(f+1, k+1, val6, SV);
        SETVALUE(f+1, k+2, val7, SV);
        SETVALUE(f+1, k+3, val8, SV);
        SETVALUE(f+2, k, val9, SV);
        SETVALUE(f+2, k+1, val10, SV);
        SETVALUE(f+2, k+2, val11, SV);
        SETVALUE(f+2, k+3, val12, SV);
        SETVALUE(f+3, k, val13, SV);
        SETVALUE(f+3, k+1, val14, SV);
        SETVALUE(f+3, k+2, val15, SV);
        SETVALUE(f+3, k+3, val16, SV);  
      }
    }    
    double time_item_update = omp_get_wtime() - start;;
    std::cout << "Time of item_update: " << time_item_update << std::endl;
	  if (showloss)
		  loss_pre = showLoss(iter, (time_user_update+time_item_update), loss_pre);
	}
}

void MF_fastALS::runOneIteration() {
	// Update user latent vectors
	for (int u = 0; u < this->userCount; u++) {
		update_user_thread(u);
	}
	// Update item latent vectors
	for (int i = 0; i < this->itemCount; i++) {
		update_item_thread(i);
	}
}

float MF_fastALS::showLoss(int iter, float time, float loss_pre) {
	clock_t end = clock();
	float loss_cur = loss();
	std::string symbol = loss_pre >= loss_cur ? "-" : "+";
	std::cout << "Iter=" << iter << " " <<time << " " << symbol << " loss:" << loss_cur << " " <<(float)(clock() - end)/ CLOCKS_PER_SEC << std::endl;
	return loss_cur;
}

float MF_fastALS::loss() {
	float L = reg * (U.squaredSum() + V.squaredSum());
	int i;
	for (int u = 0; u < userCount; u++) {
		float l = 0;
		int *itemList;
		itemList = trainMatrix.rows[u].spv_in;
		int size_item = trainMatrix.rows[u].n;
		for (int j = 0; j < size_item; j++) {
			i = itemList[j];
			float pred = predict(u, i);
			l += W.rows[u].spv_do[j] * pow(trainMatrix.rows[u].spv_do[j] - pred, 2);
			l -= Wi[i] * pow(pred, 2);
		}
		l += SV.mult(U.row_fal(u)).inner(U.row_fal(u));
		L += l;
    //std::cout<<L<<endl;
	}
	return L;
}

float MF_fastALS::predict(int u, int i) {
	float * u_tmp = U.matrix[u];
  float * v_tmp = V.matrix[i];
  float res = 0;
  //int len = U.numColumns;
  for(int k=0; k<factors; k++){
    res += (*(u_tmp+k)) * (*(v_tmp+k));
  }
  return res;
}

void MF_fastALS::updateModel(int u, int i) {
	trainMatrix.setValue(u, i, 1);
  W.setValue(u, i, w_new);
  if (Wi[i] == 0) { // an new item
    Wi[i] = w0 / itemCount;
    // Update the SV cache
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        float val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * Wi[i];
        SV.set(f, k, val);
        SV.set(k, f, val);
      }
    }
  }
  int maxIterOnline = 10;
  for (int iter = 0; iter < maxIterOnline; iter++) {
    update_user_thread(u);
    update_item_thread(i);
  }
}

void MF_fastALS::update_user_thread(int u){
  float ifv; 
  int *itemList;
	itemList = trainMatrix.rows[u].spv_in;
	int size_item = trainMatrix.rows[u].n;
	int size_avx = size_item / NUM_float;
	int remain = size_item % NUM_float;
  if (size_item == 0)        return ;    // user has no ratings
  int i;
  float test[size_avx];
  float *prediction_items = new float[size_item];
  float *rating_items = new float[size_item]; 
  float *w_items = new float[size_item]; 
    
  //one column of the V matrix
  float *v_col = new float[size_item];

  // prediction cache for the user 
  for (int j = 0; j < size_item; j++) {
    i = itemList[j];
    prediction_items[j] = predict(u, i);
    rating_items[j] = trainMatrix.rows[u].spv_do[j];
    w_items[j] = W.rows[u].spv_do[j];
  }

  //DenseVec oldVector = U.row(u);
  float *uget = U.matrix[u];
  float *numer_tmp = new float[NUM_float];
  //float *vget = V.matrix[u];
  for (int f = 0; f < factors; f++) {
    float numer = 0, denom = 0;
    // O(K) complexity for the negative part
    float *svget = SV.matrix[f];
    for(int j = 0; j<size_item; j++){
      i = itemList[j];
      v_col[j] = V.matrix[i][f];
    }
    for(int k = 0; k<factors; k++){                                         
      if(k!=f)
        numer -= (*(uget+k)) * (*(svget+k));
    }   
    float ufget = U.matrix[u][f];
    float mius = -1;

    for (int j = 0; j<size_item; j++) {
      i = itemList[j];
      ifv = *(v_col+j);
      prediction_items[j] -= ufget * ifv;
      numer += (w_items[j] * rating_items[j] - (w_items[j] - Wi[i]) * prediction_items[j]) * ifv;
      denom += (w_items[j] - Wi[i]) * ifv * ifv;
    }
    denom +=(*(svget+f)) + reg;
      
    // Parameter Update
    (*(uget+f)) = numer / denom;
    float tmp_uget = numer / denom;
     
    // Update the prediction cache
    for (int j = 0; j<size_item; j++){                                                      
      prediction_items[j] += tmp_uget * v_col[j];
    }
  } // end for f

  delete [] prediction_items;
  delete [] rating_items;
  delete [] w_items;
  delete [] numer_tmp;
  delete [] v_col;
}

void MF_fastALS::update_user_SU(float *oldVector, float *uget){
  float ** suget = SU.matrix;
  float * sugetu;
  for (int f = 0; f < factors; f++) {
      sugetu = *( suget + f);
      for (int k = 0; k <= f; k++) {
        float val = (*(sugetu + k)) - (*(oldVector+f)) * (*(oldVector+k)) + (*(uget+f)) * (*(uget+k));
        SU.set(f, k, val);
        SU.set(k, f, val);
      }
    }
}


void  MF_fastALS::update_item_thread(int i){
  float ifu;
  int *userList = trainMatrix.cols[i].spv_in;
  int size_user = trainMatrix.cols[i].n;
  int size_avx = size_user / NUM_float;
  int remain = size_user % NUM_float;
    
  if (size_user == 0)        return; // item has no ratings.
  int u;
  int wii = Wi[i];
  float test[size_avx];
  float *prediction_users = new float[size_user];
  float *rating_users = new float[size_user];
  float *w_users = new float[size_user];
  float *u_col = new float[size_user];
  // prediction cache for the item
  for (int j = 0; j < size_user; j++) {
    u = userList[j];
    prediction_users[j] = predict(u, i);
    rating_users[j] = trainMatrix.cols[i].spv_do[j];
    w_users[j] = W.cols[i].spv_do[j];
  }
  float *vget = V.matrix[i];
  float *numer_tmp = new float[NUM_float];

  //DenseVec oldVector = V.row(i);
  for (int f = 0; f < factors; f++) {
    // O(K) complexity for the w0 part
    float numer = 0, denom = 0;
    float *suget = SU.matrix[f];
    for(int j = 0; j<size_user; j++){
      u = userList[j];
      u_col[j] = U.matrix[u][f];
    }
      
    for (int k = 0; k < factors; k++) {
      if (k != f)
        numer -= (*(vget+k)) * (*(suget+k));
    }
    numer *= Wi[i];
    float ifget = V.matrix[i][f];
    float mius = -1;
    for (int j=0; j<size_user; j++) {
      u = userList[j];
      ifu = *(u_col+j);
      prediction_users[j] -= ifu * ifget;
      numer += (w_users[j] * rating_users[j] - (w_users[j] - Wi[i]) * prediction_users[j]) * ifu;
      denom += (w_users[j] - Wi[i]) * ifu * ifu;
    }
    denom += Wi[i] * (*(suget+f)) + reg;
      
    // Parameter update
    (*(vget+f)) = numer / denom;
    float tmp_vget = numer / denom;
      
    // Update the prediction cache for the item
    for(int j=0; j<size_user; j++)
      prediction_users[j] += tmp_vget * u_col[j];
  } // end for f

  delete [] prediction_users;
  delete [] rating_users;
  delete [] w_users;
  delete [] u_col;
  delete [] numer_tmp;
}

void MF_fastALS::update_item_SV(int i, float *oldVector, float *vget){
  float ** svget = SV.matrix;
  float * svgeti;
  for (int f = 0; f < factors; f++) {
    svgeti = *(svget + f);  
    for (int k = 0; k <= f; k++) {
      float val = (*(svgeti + k)) - (*(oldVector+f)) * (*(oldVector+k)) * Wi[i] + (*(vget+f)) * (*(vget+k)) * Wi[i];
      SV.set(f, k, val);
      SV.set(k, f, val);
    }
  }
}

void MF_fastALS::initS() {
  SU = U.transpose().mult(U);
  SV= DenseMat(factors, factors);
  for (int f = 0; f < factors; f++) {
    for (int k = 0; k <= f; k++) {
      float val = 0;
      for (int i = 0; i < itemCount; i++)
        val += V.get(i, f) * V.get(i, k) * Wi[i];
      SV.set(f, k, val);
      SV.set(k, f, val);
    }
  }
}

float MF_fastALS::getHitRatio(std::vector<int> rankList, int gtItem) {
    for (int item : rankList) {
      if (item == gtItem)    return 1;
    }
    return 0;
  }
float MF_fastALS::getNDCG(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
      int item = rankList[i];
      if (item == gtItem)
        return log(2) / log(i + 2);
    }
    return 0;
  }
float MF_fastALS::getPrecision(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
      int item = rankList[i];
      if (item == gtItem)
        return 1.0 / (i + 1);
    }
    return 0;
  }

std::vector<float> MF_fastALS::evaluate_for_user(int u, int gtItem, int topK) {
  std::vector<float> result(3);
  std::map<int, float> map_item_score;
  float maxScore;
  maxScore = predict(u, gtItem);
  int countLarger = 0;
  for (int i = 0; i < itemCount; i++) {
    float score = predict(u, i);
    map_item_score.insert(std::make_pair(i, score));
    if (score > maxScore) countLarger++;
    if (countLarger > topK)  return result;
  }
  std::vector<int> rankList;
  std::vector<std::pair<int, float>>top_K(topK);
  std::partial_sort_copy(map_item_score.begin(), map_item_score.end(), top_K.begin(), top_K.end(), [](std::pair<const int, int> const& l, std::pair<const int, int> const& r){
    return l.second > r.second;
  });
  for (auto const& p : top_K){
    rankList.push_back(p.first);
  }
  result[0] = getHitRatio(rankList, gtItem);
  result[1] = getNDCG(rankList, gtItem);
  result[2] = getPrecision(rankList, gtItem);
  return result;
}

float MF_fastALS::Calculate_RMSE(){
  float L = reg * (U.squaredSum() + V.squaredSum());
  int i;
  for (int u = 0; u < userCount; u++) {
    float l = 0;
    i = testRatings[u].itemId;
    float score = testRatings[u].score;
    float pred = predict(u, i);
    l += W.getValue(u, i) * pow(score - pred, 2);
    l -= Wi[i] * pow(pred, 2);
    l += SV.mult(U.row_fal(u)).inner(U.row_fal(u));
    L += l;    
  }
  return L / userCount ;
}

MF_fastALS::~MF_fastALS(){
  delete [] Wi;
}
