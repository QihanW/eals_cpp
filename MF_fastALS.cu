#include "omp.h"
#include "MF_fastALS.cuh"
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
//#include <immintrin.h>
#include "DenseVec.h"
#include "DenseMat.h"
#include "SparseVec.h"
#include "SparseMat.h"
#include "Rating.h"

#define NUM_float 8
#define TILE_SIZE 8
#define BLOCK_NUM 128
#define THREAD_NUM 256

MF_fastALS::MF_fastALS(SparseMat trainMatrix1, std::vector<Rating> testRatings1,
	int topK1, int threadNum1, int factors1, int maxIter1, float w01, float alpha1, float reg1,
	float init_mean1, float init_stdev1, bool showProgress1, bool showLoss1, int userCount1,
	int itemCount1)
{
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

__global__ void computeSUValue (float *u_clone, float *u, int f, int k, int factors,  float *result, int userCount){
  __shared__ float cache[THREAD_NUM];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  int index;
  for(int i = bid*THREAD_NUM+tid; i < userCount; i+=BLOCK_NUM*THREAD_NUM){
    index = i * factors;
    temp = temp - u_clone[index+f] * u_clone[index+k] + u[index+f] * u[index+k];
  }

  cache[cacheIndex] = temp;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0)
    result[blockIdx.x] = cache[0];

}

__global__ void computeSVValue (float *u_clone, float *u, int f, int k, int factors, float *result, int userCount, float *wi_cu){
  __shared__ float cache[THREAD_NUM];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  float val = 0;
  int index;
  for(int i = bid*THREAD_NUM+tid; i < userCount; i+=BLOCK_NUM*THREAD_NUM){
    index = i * factors;
    temp = 0 - u_clone[index+f] * u_clone[index+k] + u[index+f] * u[index+k];
    val += temp * wi_cu[i];
  }

  cache[cacheIndex] = val;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0)
    result[blockIdx.x] = cache[0];

}

void MF_fastALS::buildModel() {
	omp_set_num_threads(16);
  float loss_pre = FLT_MAX;
  for (int iter = 0; iter < maxIter; iter++) {
		DenseMat u_clone(userCount, factors);
		#pragma omp parallel for
		for (int i = 0; i<userCount; i++){
      for (int j = 0; j<factors; j++){
        u_clone.matrix[i][j] = U.matrix[i][j];
      }
    }
 
		double start = omp_get_wtime();
    
		updateUserSchedule();
		double end1 = omp_get_wtime();
    std::cout<<"first part of user "<<(end1-start)<<std::endl;
  
    int byteSize = sizeof(float)*userCount*factors;
    float *U_clone_cu, *U_cu, *U_h, *U_clone_h, *result, *result_cu;
    U_h = (float *)malloc(byteSize);
    U_clone_h = (float *)malloc(byteSize);
    result = (float *)malloc(sizeof(float)*(BLOCK_NUM));
    cudaMalloc((void**)&U_clone_cu, byteSize);
    cudaMalloc((void**)&U_cu, byteSize);
    cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));
    int ii = 0;

    #pragma omp parallel for
    for (int u = 0; u < userCount; u++){
      for(int j = 0; j < factors; j++){
        U_h[ii] = U.matrix[u][j];
        U_clone_h[ii] = u_clone.matrix[u][j];
        ii++;
      }
    }

    cudaMemcpy(U_clone_cu, U_clone_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(U_cu, U_h, byteSize, cudaMemcpyHostToDevice);
    
    //int indexf, indexk;
    for (int f = 0; f < factors; f++) { 
      for (int k = 0; k <= f; k++) {                                          
        float val = SU.matrix[f][k];
        computeSUValue<<<BLOCK_NUM,THREAD_NUM,0>>>(U_clone_cu, U_cu, f, k, factors, result_cu, userCount);
        cudaMemcpy(result, result_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+:val)                          
        for (int u = 0; u < BLOCK_NUM; u++){                                  
          val += result[u];                                                       
        }                                                                     
        SU.matrix[f][k] = val;
        SU.matrix[k][f] = val;
       }                                                                  
    }

    cudaFree(U_clone_cu);
    cudaFree(U_cu);
    cudaFree(result_cu);
    cudaFreeHost(U_clone_h);
    cudaFreeHost(U_h);
    cudaFreeHost(result);

    double time_user_update = omp_get_wtime() - start;
    std::cout << "Time of user_update: " <<time_user_update<< std::endl;
    
    /*
    float tmp;
    for (int f = 0; f < factors; f++) {                                                                                           
	    for (int k = 0; k <= f; k++) {
	      float val = SU.matrix[f][k];
	      #pragma omp parallel for reduction(+:val) 
	      for (int u = 0; u < userCount; u++){
		      tmp = 0 - u_clone.matrix[u][f] * u_clone.matrix[u][k] + U.matrix[u][f] * U.matrix[u][k];
          val += tmp;
        }
		    SU.matrix[f][k] = val;                                                      
		    SU.matrix[k][f] = val;                                                                                                                                 
      }     
    }
    double time_user_update = omp_get_wtime() - start;
    std::cout << "Time of user_update: " <<time_user_update<< std::endl;
    */

	  DenseMat v_clone(itemCount, factors);
    #pragma omp parallel for 
    for (int i = 0; i<itemCount; i++){
	    for (int j = 0; j<factors; j++){
		    v_clone.matrix[i][j] = V.matrix[i][j];
	    }
    }
    start = omp_get_wtime();
		#pragma omp parallel for  shared(trainMatrix, W, U, SU, V, Wi)
		for (int i = 0; i < itemCount; i++) {
      update_item_thread(i);
    }
    //std::cout<<"First part in updating itme"<<omp_get_wtime()-start<<std::endl;
    
    byteSize = sizeof(float)*itemCount*factors;
    float *wii, *V_clone_cu, *V_cu, *V_h, *V_clone_h, *resultv, *resultv_cu, *wi_cu;
    V_h = (float *)malloc(byteSize);
    V_clone_h = (float *)malloc(byteSize);
    resultv = (float *)malloc(sizeof(float)*(BLOCK_NUM));
    cudaMalloc((void**)&V_clone_cu, byteSize);
    cudaMalloc((void**)&V_cu, byteSize);
    cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
    cudaMalloc((void**)&resultv_cu, sizeof(float)*(BLOCK_NUM));
    ii = 0;
    wii = (float *)malloc(sizeof(float)*itemCount);

    #pragma omp parallel for
    for (int u = 0; u < itemCount; u++){
      for(int j = 0; j < factors; j++){
        V_h[ii] = V.matrix[u][j];
        V_clone_h[ii] = v_clone.matrix[u][j];
        ii++;
      }
      wii[u] = Wi[u];
    }

    cudaMemcpy(V_clone_cu, V_clone_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(V_cu, V_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(wi_cu, wii, sizeof(float)*itemCount, cudaMemcpyHostToDevice);

    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        float val = SV.matrix[f][k];
        computeSVValue<<<BLOCK_NUM,THREAD_NUM,0>>>(V_clone_cu, V_cu, f, k, factors, resultv_cu, itemCount, wi_cu);
        cudaMemcpy(resultv, resultv_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+:val)
        for (int u = 0; u < BLOCK_NUM; u++){
          val += resultv[u];
        } 
        SV.matrix[f][k] = val;
        SV.matrix[k][f] = val;
      }
    }
    cudaFree(V_clone_cu);
    cudaFree(V_cu);
    cudaFree(resultv_cu);
    cudaFree(wi_cu);
    free(wii);
    free(V_h);
    free(V_clone_h);
    free(resultv);

    double time_item_update = omp_get_wtime() - start;;
    std::cout << "Time of item_update: " << time_item_update << std::endl;
    /*
    float tmp;
    for (int f = 0; f < factors; f++) {
	    for (int k = 0; k <= f; k++) {
		    float val = SV.matrix[f][k];
 		    #pragma omp parallel for reduction(+:val)
  		  for (int u = 0; u < itemCount; u++){
 			    tmp = 0 - v_clone.matrix[u][f] * v_clone.matrix[u][k] + V.matrix[u][f] * V.matrix[u][k] ;
			    tmp = tmp *  Wi[u];
			    val += tmp;
		    }
        SV.matrix[f][k] = val;
        SV.matrix[k][f] = val;
      }
    }
    double time_item_update = omp_get_wtime() - start;;
    std::cout << "Time of item_update: " << time_item_update << std::endl;
    */
		// Show loss
		if (showloss)
			loss_pre = showLoss(iter, (time_user_update+time_item_update), loss_pre);

	} // end for iter
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
	}

	return L;
}

float MF_fastALS::predict(int u, int i) {
	float * u_tmp = U.matrix[u];
  float * v_tmp = V.matrix[i];
  float res = 0;
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

void updateUserCpp(float *prediction_items, float *rating_items, float *w_items, int userCount, int factors, float reg, float **w_cu1, float **u_cu, float **v_cu, float *wi_cu, float **sv_cu, float **train_spvdo_cu, int *train_n_cu, int **train_spvin_cu, float *v_col){
  int size_item;                                                              
  float ifv, ufget, tmp_uget;                                                 
  int i;                                                                      
  float res;                                                                  
  float numer, denom;                                                         
  int *itemList;                                                              
  for(int u = 0; u < userCount; u++){                                         
    itemList = train_spvin_cu[u];                                             
    size_item = train_n_cu[u];                                                
    if (size_item == 0)        continue ;                                   
    for (int j = 0; j < size_item; j++) {                                     
      i = itemList[j];                                                        
      res = 0;                                                                
      for(int k=0; k<factors; k++){                                           
        res += u_cu[u][k] * v_cu[i][k];                         
      }                                                                       
      prediction_items[j] = res;                                              
      rating_items[j] = train_spvdo_cu[u][j];                                 
      w_items[j] = w_cu1[u][j];                                               
    }                              
    for (int f = 0; f < factors; f++) {                                       
      numer = 0, denom = 0;                                                   
      for(int j = 0; j<size_item; j++){                                       
        i = itemList[j];                                                      
        v_col[j] = v_cu[i][f];                                         
      }                                                                     
      for(int k = 0; k<factors; k++){                                         
        numer -= u_cu[u][k] * sv_cu[f][k];                        
      }                                                                         
      ufget = u_cu[u][f];                                                
      for (int j = 0; j<size_item; j++) {                                       
        i = itemList[j];                                                        
        ifv = *(v_col+j);                                                       
        prediction_items[j] -= ufget * ifv;                                     
        numer += (w_items[j] * rating_items[j] - (w_items[j] - wi_cu[i]) * prediction_items[j]) * ifv;
        denom += (w_items[j] - wi_cu[i]) * ifv * ifv;                           
      }

      denom +=sv_cu[f][f] + reg;                                         
      u_cu[u][f] = numer / denom;                                        
      tmp_uget = numer / denom;                                                 
      for (int j = 0; j<size_item; j++){                                        
        prediction_items[j] += tmp_uget * v_col[j];                             
      }                                                                        
    }                                                          
  }                                                                           
}   
/*
void MF_fastALS::updateUserSchedule(){

  int max_size = 52000, size;
  float *prediction_items;
  float *rating_items;
  float *w_items, *v_col;
  prediction_items = (float *)malloc(sizeof(float)*max_size);
  rating_items = (float *)malloc(sizeof(float)*max_size);
  w_items = (float *)malloc(sizeof(float)*max_size);
  v_col = (float *)malloc(sizeof(float)*max_size);

  float **w_cu, **u_cu, **v_cu, **sv_cu, **train_spvdo_cu;
  int **train_spvin_cu;
  w_cu = (float **)malloc(sizeof(float *)*userCount);
  u_cu = (float **)malloc(sizeof(float *)*userCount);
  v_cu = (float **)malloc(sizeof(float *)*itemCount);
  sv_cu = (float **)malloc(sizeof(float *)*factors);
  train_spvin_cu = (int **)malloc(sizeof(int *)*userCount);
  train_spvdo_cu = (float **)malloc(sizeof(float *)*userCount);

  int *train_n_cu;
  train_n_cu = (int *)malloc(sizeof(int)*userCount);

  for (int u = 0; u < userCount; u++){
    size = trainMatrix.rows[u].n;
    w_cu[u] = (float *)malloc(sizeof(float)*size);
    train_spvin_cu[u] = (int *)malloc(sizeof(int)*size);
    train_spvdo_cu[u] = (float *)malloc(sizeof(float)*size);
    for(int i = 0; i < size; i++){
      w_cu[u][i] = W.rows[u].spv_do[i];
      train_spvin_cu[u][i] = trainMatrix.rows[u].spv_in[i];
      train_spvdo_cu[u][i] = trainMatrix.rows[u].spv_do[i];
    }
    train_n_cu[u] = size;
    u_cu[u] = (float *)malloc(sizeof(float)*factors);
    for(int i = 0; i < factors; i++)
      u_cu[u][i] = U.matrix[u][i];
  }
  for (int u = 0; u < itemCount; u++){
    v_cu[u] = (float *)malloc(sizeof(float)*factors);
    for(int i = 0; i < factors; i++)
      v_cu[u][i] = V.matrix[u][i];
  }
  for (int u = 0; u < factors; u++){
    sv_cu[u] = (float *)malloc(sizeof(float)*factors);
    for(int i = 0; i < factors; i++)
      sv_cu[u][i] = SV.matrix[u][i];
  }

  //updateUserCpp(prediction_items, rating_items, w_items, userCount, factors, reg, w_cu, u_cu, v_cu, Wi, sv_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, v_col);

  for (int u = 0; u < userCount; u++){
    for(int f = 0; f < factors; f++){
      U.matrix[u][f] = u_cu[u][f];
    }
  }

  free(prediction_items);
  free(rating_items);
  free(w_items);
  free(v_col);
  free(train_n_cu);
  for (int u = 0; u < userCount; u++){
    free(w_cu[u]);
    free(train_spvin_cu[u]);
    free(train_spvdo_cu[u]);
    free(u_cu[u]);
  }
  for (int u = 0; u < itemCount; u++){
    free(v_cu[u]);
  }
  for (int u = 0; u < factors; u++){
    free(sv_cu[u]);
  }
  free(w_cu);
  free(u_cu);
  free(v_cu);
  free(sv_cu);
  free(train_spvin_cu);
  free(train_spvdo_cu);

}
*/

__global__ void updateUserCuda(float *prediction_items,  float *v_col, int uborder, int vborder, int userCount, int factors, float reg, float *w_cu, float *uvsv_cu, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_item;
  float ifv, ufget, tmp_uget;
  int i;
  float res;
  __shared__ float numer[THREAD_NUM], denom[THREAD_NUM];
  int size2;
  int index;
  double ww, rating;
  for(int u = bid; u < userCount; u+=BLOCK_NUM){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    if (size_item == 0)        continue ;
    for (int j = tid; j < size_item; j+=THREAD_NUM) {
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      for(int k=0; k<factors; k++){
        res += uvsv_cu[u*factors+k] * uvsv_cu[uborder+i*factors+k];
      }
      prediction_items[index] = res;
    }
    for (int f = 0; f < factors; f++) {
      numer[tid] = 0, denom[tid] = 0;
      ufget = uvsv_cu[u*factors+f];
      for (int j = tid; j < size_item; j+=THREAD_NUM) {
        index = size2+j;
        i = train_spvin_cu[index];
        ifv = uvsv_cu[uborder+i*factors+f];
        prediction_items[index] -= ufget * ifv;
        ww = w_cu[index];
        numer[tid] += (ww * train_spvdo_cu[index] - (ww - wi_cu[i]) * prediction_items[index]) * ifv;
        denom[tid] += (ww - wi_cu[i]) * ifv * ifv;
      }
      __syncthreads();
      i = blockDim.x/2;
      while (i != 0) {
        if (tid < i){
          numer[tid] += numer[tid + i];
          denom[tid] += denom[tid + i];
        }  
        __syncthreads();
        i /= 2;
      }
      if(tid == 0) {
        for(int k = 0; k<factors; k++){
          numer[0] -= uvsv_cu[u*factors+k] * uvsv_cu[vborder+f*factors+k];
        }
        denom[0] += uvsv_cu[vborder+f*factors+f] + reg;
        uvsv_cu[u*factors+f] = numer[0] / denom[0];
      }
      __syncthreads();
      
      tmp_uget = uvsv_cu[u*factors+f];
      for (int j = tid; j < size_item; j+=THREAD_NUM){
        i = train_spvin_cu[size2+j];
        prediction_items[size2+j] += tmp_uget * uvsv_cu[uborder+i*factors+f];
      }
      __syncthreads();
    }
  }
}

//change the data structure
 void MF_fastALS::updateUserSchedule(){
  int size, size2;
  float *prediction_items, *v_col;
  int total_size = trainMatrix.itemCount()+10;
  cudaMalloc((void**)&prediction_items,sizeof(float)*total_size);
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  float *w_cu, *uvsv_cu, *train_spvdo_cu;
  int *train_spvin_cu;

  int uvsvSize = sizeof(float) * (userCount + itemCount + factors) * factors;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&uvsv_cu, uvsvSize);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*userCount);
  float *w_h, *uvsv_h, *train_spvdo_h;
  int *train_spvin_h;
  
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&uvsv_h, uvsvSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*userCount, cudaHostAllocDefault);
  
  train_n[0] = 0;
  for (int u = 0; u < userCount; u++){
    size = trainMatrix.rows[u].n;
    size2 = train_n[u];
    train_n[u+1] = size2 + size;

    for(int i=0; i<size; i++){
      w_h[size2+i] = W.rows[u].spv_do[i];
      train_spvin_h[size2+i] = trainMatrix.rows[u].spv_in[i];
      train_spvdo_h[size2+i] = trainMatrix.rows[u].spv_do[i];
    }
  }

  int uborder = userCount * factors;
  int vborder = uborder + itemCount * factors;
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[u*factors+i] = U.matrix[u][i];
  }
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[uborder+u*factors+i] = V.matrix[u][i];
  }
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[vborder+u*factors+i] = SV.matrix[u][i];
  }
  cudaMemcpy(uvsv_cu, uvsv_h, uvsvSize, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*userCount, cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);
  updateUserCuda<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, v_col, uborder, vborder, userCount, factors, reg, w_cu, uvsv_cu, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaMemcpy(uvsv_h, uvsv_cu, uvsvSize, cudaMemcpyDeviceToHost);
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      U.matrix[u][i] = uvsv_h[u*factors+i];
  }

  cudaFree(prediction_items);
  cudaFree(v_col);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(uvsv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(uvsv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);
}

void MF_fastALS::update_user_thread(int u){
    
  float ifv; 
  int *itemList;
	itemList = trainMatrix.rows[u].spv_in;
	int size_item = trainMatrix.rows[u].n;
  if (size_item == 0)        return ;    // user has no ratings
  int i;
  float *prediction_items = new float[size_item];
  float *rating_items = new float[size_item]; 
  float *w_items = new float[size_item]; 
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
    // O(Nu) complexity for the positive part
    float ufget = U.matrix[u][f];
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
    // Update the prediction 
    for (int j = 0; j<size_item; j++){                                                      
      prediction_items[j] += tmp_uget * v_col[j];
    }
  } // end for f
  delete [] prediction_items;
  delete [] rating_items;
  delete [] w_items;
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
  if (size_user == 0)        return; // item has no ratings.
  // prediction cache for the item
  int u;
  int wii = Wi[i];
  float *prediction_users = new float[size_user];
  float *rating_users = new float[size_user];
  float *w_users = new float[size_user];
  float *u_col = new float[size_user];

  for (int j = 0; j < size_user; j++) {
    u = userList[j];
    prediction_users[j] = predict(u, i);
    rating_users[j] = trainMatrix.cols[i].spv_do[j];
    w_users[j] = W.cols[i].spv_do[j];
  }

  float *vget = V.matrix[i];
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
    // O(Ni) complexity for the positive ratings part
    float ifget = V.matrix[i][f];
    for (int j=0; j<size_user; j++) {
      u = userList[j];
      ifu = *(u_col+j);
      prediction_users[j] -= ifu * ifget;
      numer += (w_users[j] * rating_users[j] - (w_users[j] - wii) * prediction_users[j]) * ifu;
      denom += (w_users[j] - wii) * ifu * ifu;
    }
    denom += Wi[i] * (*(suget+f)) + reg;
    // Update the prediction cache for the item
    (*(vget+f)) = numer / denom;
    float tmp_vget = numer / denom;
      
    for(int j=0; j<size_user; j++)
      prediction_users[j] += tmp_vget * u_col[j];
      
    } // end for f

   delete [] prediction_users;
   delete [] rating_users;
   delete [] w_users;
   delete [] u_col;
}

void MF_fastALS::update_item_SV(int i, float *oldVector, float *vget){
  float ** svget = SV.matrix;
  float * svgeti;
  
  for (int f = 0; f < factors; f++) {
    svgeti = *(svget + f);  
    for (int k = 0; k <= f; k++) {
        float val = (*(svgeti + k)) - (*(oldVector+f)) * (*(oldVector+k)) * Wi[i]
          + (*(vget+f)) * (*(vget+k)) * Wi[i];
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
    std::partial_sort_copy(map_item_score.begin(),
                 map_item_score.end(),
                         top_K.begin(),
                         top_K.end(),
                         [](std::pair<const int, int> const& l,
                       std::pair<const int, int> const& r){
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
