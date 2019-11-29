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
#define BLOCK_NUM 2048
#define THREAD_NUM 64

MF_fastALS::MF_fastALS(SparseMat trainMatrix1, std::vector<Rating> testRatings1,
	int topK1, int threadNum1, int factors1, int maxIter1, float w01, float alpha1, float reg1,
	float init_mean1, float init_stdev1, bool showProgress1, bool showLoss1, int userCount1,
	int itemCount1, int update_index[])
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
  /*
  cudaHostAlloc((void**)&u_values, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_values, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  for (int i=0; i<userCount; i++){
    for (int j=0; j<factors; j++){
      u_values[i*factors+j] = U.matrix[i][j];
    }
  }
  for (int i=0; i<itemCount; i++){
    for (int j=0; j<factors; j++){
      v_values[i*factors+j] = V.matrix[i][j];
    }
  }
  */

  partition_index = new int[BLOCK_NUM+1];
  for(int i=0; i<=BLOCK_NUM; i++){
    partition_index[i] = update_index[i];
  }
  
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

__global__ void computeSUValue (float *u_clone, float *u, size_t pitch_u, int f, int k, int factors,  float *result, int userCount){
  __shared__ float cache[THREAD_NUM];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int cacheIndex = threadIdx.x;
  float *u_row;
  float temp = 0;
  int index;
  for(int i = bid*THREAD_NUM+tid; i < userCount; i+=BLOCK_NUM*THREAD_NUM){
    index = i * factors;
    u_row = (float*)((char*)u + i*pitch_u);
    temp = temp - u_clone[index+f] * u_clone[index+k] + u_row[f] * u_row[k];
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
	omp_set_num_threads(32);
  float loss_pre = FLT_MAX;
  for (int iter = 0; iter < maxIter; iter++) {
		
		double start = omp_get_wtime();
    
    //updateUserSchedule1();
    updateUserSchedule2();
		//double end1 = omp_get_wtime();
    //std::cout<<"first part of user "<<(end1-start)<<std::endl;

    double time_user_update = omp_get_wtime() - start;
    std::cout << "\nTime of user_update: " <<time_user_update<< std::endl;

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
    //std::cout<<"First part in updating item"<<omp_get_wtime()-start<<std::endl;
    //updateItemSchedule();

    /*
    int byteSize = sizeof(float)*itemCount*factors;
    float *wii, *V_clone_cu, *V_cu, *V_h, *V_clone_h, *resultv, *resultv_cu, *wi_cu;
    V_h = (float *)malloc(byteSize);
    V_clone_h = (float *)malloc(byteSize);
    resultv = (float *)malloc(sizeof(float)*(BLOCK_NUM));
    cudaMalloc((void**)&V_clone_cu, byteSize);
    cudaMalloc((void**)&V_cu, byteSize);
    cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
    cudaMalloc((void**)&resultv_cu, sizeof(float)*(BLOCK_NUM));
    int ii = 0;
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
    */
    double time_item_update = omp_get_wtime() - start;;
    std::cout << "Time of item_update: " << time_item_update << std::endl;
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

//basic shared memory version
__global__ void updateUserCuda_basic(float *prediction_items, float *v_col, int userEnd, int factors, float reg, float *w_cu, float *u_cu, size_t pitch_u, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_item;
  float ifv, ufget;
  int i;
  float res;
  __shared__ float numer[THREAD_NUM], denom[THREAD_NUM];
  int size2;
  int index;
  double ww;
  float *u_row, *v_row, *sv_row, *pre_row;
  for(int u = bid; u < userEnd; u+=256){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    u_row = (float*)((char*)u_cu + u*pitch_u);
    for (int j = tid; j < size_item; j+=THREAD_NUM) {
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      v_row = (float*)((char*)v_cu + i*pitch_v);
      for(int k=0; k<factors; k++){
        res += u_row[k] * v_row[k];
      }
      prediction_items[index] = res;
    }
    __syncthreads();
    for (int f = 0; f < factors; f++) {
      numer[tid] = 0, denom[tid] = 0;
      ufget = u_row[f];
      sv_row = (float*)((char*)sv_cu + f*pitch_sv);
      for (int j = tid; j < size_item; j+=THREAD_NUM) {
        index = size2+j;
        i = train_spvin_cu[index];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv = v_row[f];
        prediction_items[index] -= ufget * ifv;
        ww = w_cu[index];
        numer[tid] += (ww * train_spvdo_cu[index] - (ww - wi_cu[i]) * pre_row[j]) * ifv;
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
          numer[0] -= u_row[k] * sv_row[k];
        }
        numer[0] += u_row[f] * sv_row[f];
        denom[0] += sv_row[f] + reg;
        u_row[f] = numer[0] / denom[0];
      }
      __syncthreads();
      
      ufget = u_row[f];
      for (int j = tid; j < size_item; j+=THREAD_NUM){
        i = train_spvin_cu[size2+j];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        prediction_items[index] += ufget * v_row[f];
      }
      __syncthreads();
    }
  }
}

/*
//change the data structure
 void MF_fastALS::updateUserSchedule(){
  int size, size2;
  float *prediction_items, *v_col;
  int total_size = trainMatrix.itemCount()+10;
  cudaMalloc((void**)&prediction_items,sizeof(float)*total_size);
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  //save the old U
  float *u_clone, *u_clone_cu;
  cudaHostAlloc((void**)&u_clone, sizeof(float)*factors*userCount, cudaHostAllocDefault);
	#pragma omp parallel for
	for (int i = 0; i<userCount; i++){
    for (int j = 0; j<factors; j++){
      u_clone[i*factors+j] = U.matrix[i][j];
    }
  }
  cudaMalloc((void**)&u_clone_cu, sizeof(float)*userCount*factors);
  cudaMemcpy(u_clone_cu, u_clone, sizeof(float)*userCount*factors, cudaMemcpyHostToDevice);
  float *result, *result_cu;
  cudaHostAlloc((void**)&result, sizeof(float)*BLOCK_NUM, cudaHostAllocDefault);
  cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));

  float *w_cu, *u_cu, *v_cu, *sv_cu, *train_spvdo_cu;
  int *train_spvin_cu;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);

  //pitch optimization
  size_t pitch_u, pitch_v, pitch_sv;
  cudaMallocPitch((void**)&u_cu, &pitch_u, sizeof(float)*factors, userCount);
  cudaMallocPitch((void**)&v_cu, &pitch_v, sizeof(float)*factors, itemCount);
  cudaMallocPitch((void**)&sv_cu, &pitch_sv, sizeof(float)*factors, factors);
  
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*(userCount+1));

  float *w_h, *u_h, *v_h, *sv_h, *train_spvdo_h;
  int *train_spvin_h;
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_h, sizeof(float)*itemCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&sv_h, sizeof(float)*factors*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*(userCount+1), cudaHostAllocDefault);
  
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

  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      v_h[u*factors+i] = V.matrix[u][i];
  }
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      sv_h[u*factors+i] = SV.matrix[u][i];
  }

  //2D allocate
  cudaMemcpy2D(u_cu, pitch_u, u_h, sizeof(float)*factors, sizeof(float)*factors, userCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(v_cu, pitch_v, v_h, sizeof(float)*factors, sizeof(float)*factors, itemCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(sv_cu, pitch_sv, sv_h, sizeof(float)*factors, sizeof(float)*factors, factors, cudaMemcpyHostToDevice);


  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*(userCount+1), cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);
  
  //int *index_cu;
  //cudaMalloc((void**)&index_cu, sizeof(int)*BLOCK_NUM);
  //cudaMemcpy(index_cu, partition_index, sizeof(int)*BLOCK_NUM, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  updateUserCuda_basic<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, v_col, userCount, factors, reg, w_cu, u_cu, pitch_u, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout<<"Time of update u: "<<elapsedTime<<" ";

  cudaMemcpy2D(u_h, sizeof(float)*factors, u_cu, pitch_u, sizeof(float)*factors, userCount, cudaMemcpyDeviceToHost);
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      U.matrix[u][i] = u_h[u*factors+i];
  }

  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);
  //update su
  for (int f = 0; f < factors; f++) { 
    for (int k = 0; k <= f; k++) {                                          
      float val = SU.matrix[f][k];
      computeSUValue<<<BLOCK_NUM,THREAD_NUM,0>>>(u_clone_cu, u_cu, pitch_u, f, k, factors, result_cu, userCount);
      cudaMemcpy(result, result_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
      #pragma omp parallel for reduction(+:val)                          
      for (int u = 0; u < BLOCK_NUM; u++){                                  
        val += result[u];                                                       
      }                                                                     
      SU.matrix[f][k] = val;
      SU.matrix[k][f] = val;
     }                                                                  
  } 
  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  float elapsedTime2;
  cudaEventElapsedTime(&elapsedTime2, start2, stop2);
  cout<<"Time of update su: "<<elapsedTime2<<" ";

  //free update su
  cudaFreeHost(u_clone);
  cudaFree(u_clone_cu);
  cudaFreeHost(result);
  cudaFree(result_cu);

  //free update u
  cudaFree(prediction_items);
  cudaFree(v_col);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(u_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(u_h);
  cudaFreeHost(v_h);
  cudaFreeHost(sv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);

 }
 
*/

/*
//gpu+cpu basic version
__global__ void updateUserCuda(float *numer, float *denom, int userCount, int factors, float reg, float *w_cu, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu, int *index_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int size_item;
  float ifv;
  int i, index_u;
  int size2;
  int index;
  float ww;
  __shared__ float numer_sh[THREAD_NUM];
  __shared__ float denom_sh[THREAD_NUM];
  float *v_row, *sv_row;

  for(int u = bidx; u < userCount; u+=gridDim.x){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    if (size_item == 0)        continue ;
    for (int f = bidy; f < factors; f+=gridDim.y){
      index_u = u*factors+f;
      numer_sh[tidx] = 0;
      denom_sh[tidx] = 0;
      sv_row = (float*)((char*)sv_cu + f*pitch_sv);
      for (int j = tidx; j < size_item; j+=blockDim.x) {
        index = size2+j;
        i = train_spvin_cu[index];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv = v_row[f];
        ww = w_cu[index];
        numer_sh[tidx] += ww * train_spvdo_cu[index] * ifv;
        denom_sh[tidx] += (ww - wi_cu[i]) * ifv * ifv;
      }
      __syncthreads();
      i = blockDim.x/2;
      while (i != 0) {
        if (tidx < i){
          numer_sh[tidx] += numer_sh[tidx + i];
          denom_sh[tidx] += denom_sh[tidx + i];
        }  
        __syncthreads();
        i /= 2;
      }
      if(tidx == 0) {
        denom[index_u] = denom_sh[0] + sv_row[f] + reg;
        numer[index_u] = numer_sh[0];
      }
      __syncthreads();
    }
  }
}
*/
/*
__global__ void updatePrediction(float *numer, float *denom, float *prediction_items, size_t pitch_pre, int userCount, int factors, float reg, float *w_cu, float *u_cu, size_t pitch_u, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
  int tid = threadIdx.x;
  //int tidy = threadIdx.y;
  int bid = blockIdx.x;
  //int bidy = blockIdx.y;
  int size_item;
  float ifv, ufget;
  int i, index_u;
  int size2;
  int index;
  float res;
  float *u_row, *v_row, *sv_row, *pre_row;
  __shared__ float numer_sh[THREAD_NUM];
  for(int u = bid; u < userCount; u+=gridDim.x){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    u_row = (float*)((char*)u_cu + u*pitch_u);
    pre_row = (float*)((char*)prediction_items + u*pitch_pre);
    if (size_item == 0)        continue ;
    for (int j = tid; j < size_item; j+=blockDim.x) {
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      for(int k=0; k<factors; k++){
        res += u_cu[u*factors+k] * v_cu[i*factors+k];
      }
      pre_row[j] = res;
    }
    for (int f = 0; f < factors; f++) {
      index_u = u*factors+f;
      ufget = u_row[f];
      sv_row = (float*)((char*)sv_cu + f*pitch_sv);
      for (int j = tid; j < size_item; j+=blockDim.x) {
          index = size2+j;
          i = train_spvin_cu[index];
          v_row = (float*)((char*)v_cu + i*pitch_v);
          ifv = v_row[f];
          pre_row[j] -= ufget * ifv;
          numer_sh[tid] += (wi_cu[i] - w_cu[index]) * pre_row[j] * ifv;
      }
      //__syncthreads();
      
      i = blockDim.x/2;
      while (i != 0) {
        if (tid < i){
          numer_sh[tid] += numer_sh[tid + i];
        }  
        __syncthreads();
        i /= 2;
      }
      
      if(tid == 0) {
        for(int k = 0; k<factors; k++){
          numer_sh[0] -= u_row[k] * sv_row[k];
        }
        numer[index_u] = numer_sh[0] + u_row[f] * sv_row[f];
        u_row[f] = numer[index_u] / denom[index_u];
      }
      __syncthreads();
      
      ufget = u_row[f];
      for (int j = tid; j < size_item; j+=blockDim.x){
        i = train_spvin_cu[size2+j];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        pre_row[j] += ufget * v_row[f];
      }
      __syncthreads();
    }
  }
}
*/
//gpu+cpu second version
__global__ void updateUserCuda_all(float *prediction_items,  float *v_col, int uborder, int vborder, int userCount, int factors, float reg, float *w_cu, float *uvsv_cu, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
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
 void MF_fastALS::updateUserSchedule1(){
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
  
  int userBorder = 0;
  cudaMemcpy(uvsv_cu, uvsv_h, uvsvSize, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*userCount, cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  updateUserCuda_all<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, v_col, uborder, vborder, userBorder, factors, reg, w_cu, uvsv_cu, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout<<"Time of update u1: "<<elapsedTime<<" ";
  
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
//updating both numer and denom
__global__ void updateUserCuda(float *numer, float *denom, int userBegin, int userEnd, int factors, float reg, float *w_cu, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu, int *index_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  //int bidy = blockIdx.y;
  int size_item, i;
  float ifv;
  //int i, index_u;
  int size2;
  //int index;
  //float ww;
  __shared__ float numer_sh[THREAD_NUM];
  __shared__ float denom_sh[THREAD_NUM];
  __shared__ int index[THREAD_NUM];
  __shared__ int index_u_sh[THREAD_NUM];
  __shared__ float ifv_sh[THREAD_NUM];
  __shared__ float ww_sh[THREAD_NUM];
  float *v_row, *sv_row;

  for(int u = bidx+userBegin; u < userEnd; u+=BLOCK_NUM){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    //int f = tidx;s
      index_u_sh[tidx] = u*factors+tidx;
      numer_sh[tidx] = 0;
      denom_sh[tidx] = 0;
      sv_row = (float*)((char*)sv_cu + tidx*pitch_sv);
      for (int j = 0; j < size_item; j++) {
        index[tidx] = size2+j;
        i = train_spvin_cu[index[tidx]];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv_sh[tidx] = v_row[tidx];
        ww_sh[tidx] = w_cu[index[tidx]];
        numer_sh[tidx] += ww_sh[tidx] * train_spvdo_cu[index[tidx]] * ifv_sh[tidx];
        denom_sh[tidx] += (ww_sh[tidx] - wi_cu[i]) * ifv_sh[tidx] * ifv_sh[tidx];
      }
      denom[index_u_sh[tidx]] = denom_sh[tidx] + sv_row[tidx] + reg;
      numer[index_u_sh[tidx]] = numer_sh[tidx];
  }
}
/*
__global__ void updateUserCuda_dense(float *numer, float *denom, int userBegin, int userEnd, int factors, float reg, float *w_cu, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu, int *index_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int size_item, i;
  float ifv;
  //int i, index_u;
  int size2;
  //int index;
  //float ww;
  __shared__ float numer_sh[128];
  __shared__ float denom_sh[128];
  __shared__ int index[128];
  __shared__ int index_u_sh[128];
  __shared__ float ifv_sh[128];
  __shared__ float ww_sh[128];
  float *v_row, *sv_row;
  float numerr = 0;
  float denomm = 0;

  for(int u = bidx+userBegin; u < userEnd; u+=64){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
      index_u_sh[bidy] = u*factors+bidy;
      numer_sh[bidy] = 0;
      denom_sh[bidy] = 0;
      sv_row = (float*)((char*)sv_cu + bidy*pitch_sv);
      for (int j = 0; j < size_item; j++) {
        index[j] = size2+j;
        i = train_spvin_cu[index[j]];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv_sh[j] = v_row[j];
        ww_sh[j] = w_cu[index[j]];
        numerr += ww_sh[j] * train_spvdo_cu[index[j]] * ifv_sh[j];
        denomm += (ww_sh[j] - wi_cu[i]) * ifv_sh[j] * ifv_sh[j];
      }
      denom[index_u_sh[tidx]] = denom_sh[tidx] + sv_row[tidx] + reg;
      numer[index_u_sh[tidx]] = numer_sh[tidx];
  }
}
*/
//update numer
__global__ void updateUserNumer(float *numer, float *denom, int userCount, int factors, float reg, float *w_cu, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu, int *index_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  //int bidy = blockIdx.y;
  int size_item;
  float ifv;
  int i, index_u;
  int size2;
  int index;
  float ww;
  __shared__ float numer_sh[THREAD_NUM];
  //__shared__ float denom_sh[THREAD_NUM];
  float *v_row, *sv_row;

  for(int u = index_cu[bidx]; u < index_cu[bidx+1]; u++){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    int f = tidx;
      index_u = u*factors+f;
      numer_sh[tidx] = 0;
      //denom_sh[tidx] = 0;
      sv_row = (float*)((char*)sv_cu + f*pitch_sv);
      for (int j = 0; j < size_item; j++) {
        index = size2+j;
        i = train_spvin_cu[index];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv = v_row[f];
        ww = w_cu[index];
        numer_sh[tidx] += ww * train_spvdo_cu[index] * ifv;
        //denom_sh[tidx] += (ww - wi_cu[i]) * ifv * ifv;
      }
      //denom[index_u] = denom_sh[tidx] + sv_row[f] + reg;
      numer[index_u] = numer_sh[tidx];
  }
}
//update denom
__global__ void updateUserDenom(float *numer, float *denom, int userCount, int factors, float reg, float *w_cu, float *v_cu, size_t pitch_v, float *sv_cu, size_t pitch_sv, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu, int *index_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  //int bidy = blockIdx.y;
  int size_item;
  float ifv;
  int i, index_u;
  int size2;
  int index;
  float ww;
  //__shared__ float numer_sh[THREAD_NUM];
  __shared__ float denom_sh[THREAD_NUM];
  float *v_row, *sv_row;

  for(int u = index_cu[bidx]; u < index_cu[bidx+1]; u++){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    int f = tidx;
      index_u = u*factors+f;
      //numer_sh[tidx] = 0;
      denom_sh[tidx] = 0;
      sv_row = (float*)((char*)sv_cu + f*pitch_sv);
      for (int j = 0; j < size_item; j++) {
        index = size2+j;
        i = train_spvin_cu[index];
        v_row = (float*)((char*)v_cu + i*pitch_v);
        ifv = v_row[f];
        ww = w_cu[index];
        //numer_sh[tidx] += ww * train_spvdo_cu[index] * ifv;
        denom_sh[tidx] += (ww - wi_cu[i]) * ifv * ifv;
      }
      denom[index_u] = denom_sh[tidx] + sv_row[f] + reg;
      //numer[index_u] = numer_sh[tidx];
  }
}


__global__ void initialPredictionItems(int userCount, int factors, float *v_cu, size_t pitch_v, float *u_cu, size_t pitch_u, int *train_n_cu, int *train_spvin_cu, float *prediction_items){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int index, i, res, size_item, size2;
  float *u_row, *v_row;
  for(int u = bid; u < userCount; u+=BLOCK_NUM){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    u_row = (float*)((char*)u_cu + u*pitch_u);
    for (int j = tid; j < size_item; j+=THREAD_NUM){
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      v_row = (float*)((char*)v_cu + i*pitch_v);
      for(int k=0; k<factors; k++){
        res += u_row[k] * v_row[k];
      }
      prediction_items[index] = res;
    }
  }
}



void MF_fastALS::update_user_cpu(int u, float *u_numer_h, float *u_denom_h){
  float tmp_uget, ufget, ifv;
  int i, size_item;
  int *itemList;
  float *uget;
  float *svget;
  float tmp_numer, tmp_denom;
  float *prediction_items;
  float *v_col;

  //std::cout<<"u: "<<u<<" userBegin: "<<userBegin<<std::endl;
  size_item = trainMatrix.rows[u].n;
  itemList = trainMatrix.rows[u].spv_in;
  prediction_items = new float[size_item];
  v_col = new float[size_item];

  for (int j = 0; j < size_item; j++) {
      i = itemList[j];
      prediction_items[j] = predict(u, i);
  }
  uget = U.matrix[u];
  for (int f = 0; f < factors; f++) {
    ufget = U.matrix[u][f];
    svget = SV.matrix[f];
    tmp_numer = u_numer_h[(u)*factors+f];
    tmp_denom = u_denom_h[(u)*factors+f];
    for(int j = 0; j<size_item; j++){
      i = itemList[j];
      v_col[j] = V.matrix[i][f];
    }
    for(int k = 0; k<factors; k++){
      if(k!=f){
        tmp_numer -= (*(uget+k)) * (*(svget+k));
      }
    }
    for (int j = 0; j<size_item; j++) {
      i = itemList[j];
      ifv = *(v_col+j);
      prediction_items[j] -= ufget * ifv;
      tmp_numer += (Wi[i] - W.rows[u].spv_do[j]) * prediction_items[j] * ifv;
    }
    (*(uget+f)) =  tmp_numer / tmp_denom;
    tmp_uget = (*(uget+f));
    for (int j = 0; j<size_item; j++){
      prediction_items[j] += tmp_uget * v_col[j];
    }
  }
  delete [] prediction_items;
  delete [] v_col;
}
/*
void MF_fastALS::updateUserSchedule2(){
  omp_set_num_threads(64);
  
  int size, size2;
  float *prediction_items, *v_col;
  int total_size = trainMatrix.itemCount()+10;
  cudaMalloc((void**)&prediction_items,sizeof(float)*total_size);
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  //save the old U
  float *u_clone, *u_clone_cu;
  cudaHostAlloc((void**)&u_clone, sizeof(float)*factors*userCount, cudaHostAllocDefault);
	#pragma omp parallel for
	for (int i = 0; i<userCount; i++){
    for (int j = 0; j<factors; j++){
      u_clone[i*factors+j] = U.matrix[i][j];
    }
  }
  //cudaMalloc((void**)&u_clone_cu, sizeof(float)*userCount*factors);
  //cudaMemcpy(u_clone_cu, u_clone, sizeof(float)*userCount*factors, cudaMemcpyHostToDevice);
  float *result, *result_cu;
  cudaHostAlloc((void**)&result, sizeof(float)*BLOCK_NUM, cudaHostAllocDefault);
  cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));

  float *w_cu, *u_cu, *v_cu, *sv_cu, *train_spvdo_cu;
  int *train_spvin_cu;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);

  //pitch optimization
  size_t pitch_u, pitch_v, pitch_sv;
  cudaMallocPitch((void**)&u_cu, &pitch_u, sizeof(float)*factors, userCount);
  cudaMallocPitch((void**)&v_cu, &pitch_v, sizeof(float)*factors, itemCount);
  cudaMallocPitch((void**)&sv_cu, &pitch_sv, sizeof(float)*factors, factors);
  
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*(userCount+1));

  float *w_h, *u_h, *v_h, *sv_h, *train_spvdo_h;
  int *train_spvin_h;
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_h, sizeof(float)*itemCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&sv_h, sizeof(float)*factors*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*(userCount+1), cudaHostAllocDefault);
  
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

  #pragma omp parallel for
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  
  #pragma omp parallel for
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      v_h[u*factors+i] = V.matrix[u][i];
  }
  #pragma omp parallel for
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      sv_h[u*factors+i] = SV.matrix[u][i];
  }

  //2D allocate
  cudaMemcpy2D(u_cu, pitch_u, u_h, sizeof(float)*factors, sizeof(float)*factors, userCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(v_cu, pitch_v, v_h, sizeof(float)*factors, sizeof(float)*factors, itemCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(sv_cu, pitch_sv, sv_h, sizeof(float)*factors, sizeof(float)*factors, factors, cudaMemcpyHostToDevice);

  double before_kernel = omp_get_wtime();
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*(userCount+1), cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);
  
  int *index_cu;
  cudaMalloc((void**)&index_cu, sizeof(int)*(BLOCK_NUM+1));
  cudaMemcpy(index_cu, partition_index, sizeof(int)*(BLOCK_NUM+1), cudaMemcpyHostToDevice);
  
  float *u_numer, *u_denom;
  cudaMalloc((void**)&u_numer,sizeof(float)*userCount*factors);
  cudaMalloc((void**)&u_denom,sizeof(float)*userCount*factors);
  float *u_numer_h, *u_denom_h;
  cudaHostAlloc((void**)&u_numer_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_denom_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);

  int userBorder = 0;

  //test overlap function
  /*cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
  cudaGetDeviceProperties(&prop, deviceID);
  if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
	}*/
  /*
  //create streams
  cudaStream_t stream1; 
  cudaStream_t stream2; 
  cudaStreamCreate(&stream1);  
  cudaStreamCreate(&stream2);  

  int userBegin=0;
  int userEnd=20000;
  int dimx, dimy;
  dimx = factors;
  dimy = 16;
  dim3 block(64);
  dim3 grid(dimx, dimy);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //updateUserCuda<<<BLOCK_NUM, THREAD_NUM, 0, stream1>>>(u_numer, u_denom, userBegin, userEnd, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu); 
  updateUserCuda<<<BLOCK_NUM, THREAD_NUM, 0, stream2>>>(u_numer, u_denom, userBegin, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);
  //updateUserNumer<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);
  cudaMemcpy(u_numer_h, u_numer, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_denom_h, u_denom, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout<<"Time of update u2: "<<elapsedTime<<" ";
  cudaStreamDestroy(stream1);  
  cudaStreamDestroy(stream2);  
   
  //cpp version
  double start_openmp = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi, u_numer_h, u_denom_h)
  for(int u = userBorder; u < userCount; u++){
    update_user_cpu(u, u_numer_h, u_denom_h);
  }
  double time_openmp = omp_get_wtime() - start_openmp;
  std::cout<<"Openmp time: "<<time_openmp<<" ";
  
  
  #pragma omp parallel for
  for (int u = userBorder; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  

  //free update su
  
  cudaFreeHost(u_clone);
  cudaFreeHost(result);
  cudaFree(result_cu);

  //free update u
  cudaFree(u_numer);
  cudaFree(u_denom);
  cudaFree(prediction_items);
  cudaFree(v_col);
  cudaFree(index_cu);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(u_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(u_h);
  cudaFreeHost(u_numer_h);
  cudaFreeHost(u_denom_h);
  cudaFreeHost(v_h);
  cudaFreeHost(sv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);
  
 }
*/

 void MF_fastALS::updateUserSchedule2(){
  omp_set_num_threads(64);
  
  int size, size2;
  float *prediction_items, *v_col;
  int total_size = trainMatrix.itemCount()+10;
  cudaMalloc((void**)&prediction_items,sizeof(float)*total_size);
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  //cudaMalloc((void**)&u_clone_cu, sizeof(float)*userCount*factors);
  //cudaMemcpy(u_clone_cu, u_clone, sizeof(float)*userCount*factors, cudaMemcpyHostToDevice);
  float *result, *result_cu;
  cudaHostAlloc((void**)&result, sizeof(float)*BLOCK_NUM, cudaHostAllocDefault);
  cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));

  float *w_cu, *u_cu, *v_cu, *sv_cu, *train_spvdo_cu;
  int *train_spvin_cu;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);

  //pitch optimization
  size_t pitch_u, pitch_v, pitch_sv;
  cudaMallocPitch((void**)&u_cu, &pitch_u, sizeof(float)*factors, userCount);
  cudaMallocPitch((void**)&v_cu, &pitch_v, sizeof(float)*factors, itemCount);
  cudaMallocPitch((void**)&sv_cu, &pitch_sv, sizeof(float)*factors, factors);
  
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*(userCount+1));

  float *w_h, *u_h, *v_h, *sv_h, *train_spvdo_h;
  int *train_spvin_h;
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_h, sizeof(float)*itemCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&sv_h, sizeof(float)*factors*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*(userCount+1), cudaHostAllocDefault);
  
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

  #pragma omp parallel for
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  
  #pragma omp parallel for
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      v_h[u*factors+i] = V.matrix[u][i];
  }
  #pragma omp parallel for
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      sv_h[u*factors+i] = SV.matrix[u][i];
  }
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  int *index_cu;
  cudaMalloc((void**)&index_cu, sizeof(int)*(BLOCK_NUM+1));
  float *u_numer, *u_denom;
  cudaMalloc((void**)&u_numer,sizeof(float)*userCount*factors);
  cudaMalloc((void**)&u_denom,sizeof(float)*userCount*factors);
  int userBorder = 0;
  
  /*
  cudaEvent_t start3, stop3;
  cudaEventCreate(&start3);
  cudaEventCreate(&stop3);
  cudaEventRecord(start3, 0);
  updateUserCuda_basic<<<256, THREAD_NUM>>>(prediction_items, v_col, userBorder, factors, reg, w_cu, u_cu, pitch_u, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  float elapsedTime3;
  cudaEventElapsedTime(&elapsedTime3, start3, stop3);
  cout<<"Time of update u1: "<<elapsedTime3<<" ";

  cudaMemcpy2D(u_h, sizeof(float)*factors, u_cu, pitch_u, sizeof(float)*factors, userCount, cudaMemcpyDeviceToHost);
  for (int u = 0; u < userBorder; u++){
    for(int i=0; i<factors; i++)
      U.matrix[u][i] = u_h[u*factors+i];
  }
  */

  //test overlap function
  /*cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
  cudaGetDeviceProperties(&prop, deviceID);
  if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
  }*/
  
  //memcpy
  cudaMemcpy2D(u_cu, pitch_u, u_h, sizeof(float)*factors, sizeof(float)*factors, userCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(v_cu, pitch_v, v_h, sizeof(float)*factors, sizeof(float)*factors, itemCount, cudaMemcpyHostToDevice);
  cudaMemcpy2D(sv_cu, pitch_sv, sv_h, sizeof(float)*factors, sizeof(float)*factors, factors, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*(userCount+1), cudaMemcpyHostToDevice);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);
  cudaMemcpy(index_cu, partition_index, sizeof(int)*(BLOCK_NUM+1), cudaMemcpyHostToDevice);

  //create streams
  //cudaStream_t stream1; 
  //cudaStream_t stream2; 
  //cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);  
  //cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);  

  int userBegin=0;
  int userEnd=20000;
  float *u_numer_h, *u_denom_h;
  int last = userCount - 60000;
  int usercppbegin = 0;
  cudaHostAlloc((void**)&u_numer_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_denom_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  //int dimx, dimy;
  //dim3 block(64);
  //dim3 grid(dimx, dimy);
  //std::cout<<"before kernel time: "<<omp_get_wtime() - before_kernel<<" ";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  updateUserCuda<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userBegin, userEnd, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu); 
  cudaMemcpy(u_numer_h, u_numer, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_denom_h, u_denom, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  for(int i=1; i<3; i++){
    userBegin = userEnd;
    userEnd += 20000;
    updateUserCuda<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userBegin, userEnd, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu); 
    //cpp
    usercppbegin = userBegin-20000;
    #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi, u_numer_h, u_denom_h)
    for(int u = usercppbegin; u < userBegin; u++){
      update_user_cpu(u, u_numer_h, u_denom_h);
    }
    cudaMemcpy(u_numer_h, u_numer, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_denom_h, u_denom, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  }
  updateUserCuda<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userEnd, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);
  usercppbegin = userBegin;
  #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi, u_numer_h, u_denom_h)
  for(int u = usercppbegin; u < userEnd; u++){
    update_user_cpu(u, u_numer_h, u_denom_h);
  }
  cudaMemcpy(u_numer_h, u_numer, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_denom_h, u_denom, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi, u_numer_h, u_denom_h)
  for(int u = userEnd; u < userCount; u++){
    update_user_cpu(u, u_numer_h, u_denom_h);
  }
  //updateUserNumer<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout<<"Time of update u2: "<<elapsedTime<<" ";
  //cudaStreamDestroy(stream1);  
  //cudaStreamDestroy(stream2);  
  /*
  cudaEvent_t start3, stop3;
  cudaEventCreate(&start3);
  cudaEventCreate(&stop3);
  cudaEventRecord(start3, 0);
  //updateUserCuda<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);
  updateUserDenom<<<BLOCK_NUM, THREAD_NUM>>>(u_numer, u_denom, userCount, factors, reg, w_cu, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, index_cu);
  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  float elapsedTime3;
  cudaEventElapsedTime(&elapsedTime3, start3, stop3);
  cout<<"Time of update u: "<<elapsedTime3<<" ";
  */
  //std::cout<<"all time: "<<omp_get_wtime() - before_kernel<<" ";
  /*
  cudaMemcpy2D(u_h, sizeof(float)*factors, u_cu, pitch_u, sizeof(float)*factors, userCount, cudaMemcpyDeviceToHost);
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      U.matrix[u][i] = u_h[u*factors+i];
  }
  */
  
  //initial prediction items 
  //float *prediction_items_h, *prediction_items_cu, *v_col;
  //int total_size = trainMatrix.itemCount()+10;
  //cudaMalloc((void**)&prediction_items_cu, sizeof(float)*total_size);
  //cudaHostAlloc((void**)&prediction_items_h, sizeof(float)*total_size, cudaHostAllocDefault);
 /*
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);
  initialPredictionItems<<<256, 64>>>(userCount, factors, v_cu, pitch_v, u_cu, pitch_u, train_n_cu, train_spvin_cu, prediction_items_cu);
  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  float elapsedTime2;
  cudaEventElapsedTime(&elapsedTime2, start2, stop2);
  cout<<"Time of initialize prediction_items: "<<elapsedTime2<<" ";

  cudaMemcpy(prediction_items_h, prediction_items_cu, sizeof(float)*total_size, cudaMemcpyDeviceToHost);
  cudaFree(prediction_items_cu);
  */
  //std::cout<<"gpu time: "<<omp_get_wtime() - before_kernel<<" ";
  
  //cpp version
  /*
  double start_openmp = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic, 64) shared(trainMatrix, W, U, SV, V, Wi, u_numer_h, u_denom_h)
  for(int u = userBorder; u < userCount; u++){
    update_user_cpu(u, u_numer_h, u_denom_h);
  }
  double time_openmp = omp_get_wtime() - start_openmp;
  std::cout<<"Openmp time: "<<time_openmp<<" ";
  */
  /*
  #pragma omp parallel for
  for (int u = userBorder; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  
  
  //updatePrediction<<<BLOCK_NUM,THREAD_NUM>>>(u_numer, u_denom, prediction_items, pitch_pre, userCount, factors, reg, w_cu, u_cu, pitch_u, v_cu, pitch_v, sv_cu, pitch_sv, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  //std::cout<<"cpu time: "<<omp_get_wtime() - before_kernel<<" ";
  /*
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);
  //update su
  for (int f = 0; f < factors; f++) { 
    for (int k = 0; k <= f; k++) {                                          
      float val = SU.matrix[f][k];
      computeSUValue<<<BLOCK_NUM,THREAD_NUM,0>>>(u_clone_cu, u_cu, pitch_u, f, k, factors, result_cu, userCount);
      cudaMemcpy(result, result_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
      #pragma omp parallel for reduction(+:val)                          
      for (int u = 0; u < BLOCK_NUM; u++){                                  
        val += result[u];                                                       
      }                                                                     
      SU.matrix[f][k] = val;
      SU.matrix[k][f] = val;
     }                                                                  
  } 
  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  float elapsedTime2;
  cudaEventElapsedTime(&elapsedTime2, start2, stop2);
  //cout<<"Time of update su: "<<elapsedTime2<<" ";
  omp_set_num_threads(1);
  */
  //free update su
  
  //cudaFreeHost(u_clone);
  //cudaFree(u_clone_cu);
  cudaFreeHost(result);
  cudaFree(result_cu);

  //free update u
  cudaFree(u_numer);
  cudaFree(u_denom);
  cudaFree(prediction_items);
  cudaFree(v_col);
  cudaFree(index_cu);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(u_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(u_h);
  cudaFreeHost(u_numer_h);
  cudaFreeHost(u_denom_h);
  cudaFreeHost(v_h);
  cudaFreeHost(sv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);
 }
 
/*
void updateUserSchedule(){
  int size, size2;
  float *v_col;
  int total_size = trainMatrix.itemCount()+10;
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  float *w_cu, *v_cu, *sv_cu, *train_spvdo_cu;
  int *train_spvin_cu;

  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&v_cu, sizeof(float)*itemCount*factors);
  cudaMalloc((void**)&sv_cu, sizeof(float)*factors*factors);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*(userCount+1));
  float *w_h, *v_h, *sv_h, *train_spvdo_h;
  int *train_spvin_h;
  
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_h, sizeof(float)*itemCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&sv_h, sizeof(float)*factors*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*(userCount+1), cudaHostAllocDefault);

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

  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      v_h[u*factors+i] = V.matrix[u][i];
  }
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      sv_h[u*factors+i] = SV.matrix[u][i];
  }

  cudaMemcpy(v_cu, v_h, sizeof(float)*itemCount*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(sv_cu, sv_h, sizeof(float)*factors*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*(userCount+1), cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);

  float *u_numer, *u_denom;
  cudaMalloc((void**)&u_numer,sizeof(float)*userCount*factors);
  cudaMalloc((void**)&u_denom,sizeof(float)*userCount*factors);
  
  int dimx, dimy;
  dimx = factors;
  dimy = THREAD_NUM;
  dim3 block(BLOCK_NUM);
  dim3 grid(dimx, dimy);
  updateUserCuda<<<grid, block>>>(u_numer, u_denom, v_col, userCount, factors, reg, w_cu, v_cu, sv_cu, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  float *u_numer_h = (float *)malloc(sizeof(float)*userCount*factors);
  float *u_denom_h = (float *)malloc(sizeof(float)*userCount*factors);
  cudaMemcpy(u_numer_h, u_numer, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_denom_h, u_denom, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);

  float prediction_items[total_size];
  int i, index_u;
  float res, ufget, tmp_uget, ifv;
  #pragma omp parallel for shared(trainMatrix, U, V, SV, W, Wi, prediction_items)
  for(int u = 0; u < userCount; u++){
    size = trainMatrix.rows[u].n;
    size2 = train_n[u];
    for (int j = 0; j < size; j++) {
      i = trainMatrix.rows[u].spv_in[j];
      res = 0;
      for(int k=0; k<factors; k++){
        res += U.matrix[u][k] * V.matrix[i][k];
      }
      cout<<total_size<<" "<<size2+j;
      prediction_items[size2+j] = res;
    }
    for (int f = 0; f < factors; f++) {
      ufget = U.matrix[u][f];
      index_u = u*factors+f;
      for(int k = 0; k<factors; k++){
        if(k!=f){
          u_numer_h[index_u] -= U.matrix[u][k] * SV.matrix[f][k];
        }
      }
      for (int j = 0; j<size; j++) {
        i = trainMatrix.rows[u].spv_in[j];
        ifv = V.matrix[i][f];
        prediction_items[size2+j] -= ufget * ifv;
        u_numer_h[index_u] += (Wi[i] -  W.rows[u].spv_do[j]) * prediction_items[size2+j] * ifv;
      }
      U.matrix[u][f] = u_numer_h[index_u] / u_denom_h[index_u];
      tmp_uget = U.matrix[u][f];
      for (int j = 0; j<size; j++){
        i = trainMatrix.rows[u].spv_in[j];
        prediction_items[size2+j] += tmp_uget * V.matrix[i][f];
      }
    }
  }

  cudaFree(v_col);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFree(u_numer);
  cudaFree(u_denom);
  free(u_numer_h);
  free(u_denom_h);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(v_h);
  cudaFreeHost(sv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);
}

*/
//separate kernel version
/*
__global__ void updateUserCuda2(float *numer, float *denom, float *prediction_items, int userCount, int factors, float reg, float *w_cu, float *u_cu, float *v_cu, float *sv_cu, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
  int tidx = threadIdx.x;
  //int tidy = threadIdx.y;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int size_item;
  float ifv, ufget;
  int i, index_u;
  int size2;
  int index;
  float ww, res, tmp_uget;
  __shared__ float numer_sh[THREAD_NUM];
  __shared__ float denom_sh[THREAD_NUM];

  for(int u = bidx; u < userCount; u+=gridDim.x){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    if (size_item == 0)        continue ;
    for (int f = bidy; f < factors; f+=gridDim.y){
      index_u = u*factors+f;
      numer_sh[tidx] = 0;
      denom_sh[tidx] = 0;
      for (int j = tidx; j < size_item; j+=blockDim.x) {
        index = size2+j;
        i = train_spvin_cu[index];
        ifv = v_cu[i*factors+f];
        ww = w_cu[index];
        numer_sh[tidx] += ww * train_spvdo_cu[index] * ifv;
        denom_sh[tidx] += (ww - wi_cu[i]) * ifv * ifv;
      }
      __syncthreads();
      i = blockDim.x/2;
      while (i != 0) {
        if (tidx < i){
          numer_sh[tidx] += numer_sh[tidx + i];
          denom_sh[tidx] += denom_sh[tidx + i];
        }  
        __syncthreads();
        i /= 2;
      }
      if(tidx == 0) {
        denom[index_u] = denom_sh[0] + sv_cu[f*factors+f] + reg;
        numer[index_u] = numer_sh[0];
      }
      __syncthreads();
    }
  }

  int threadId = tidx + blockDim.x * (bidy * gridDim.x + bidx);
  int dist = gridDim.x * gridDim.y * blockDim.x;
  for(int u = threadId; u < userCount; u+=dist){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    if (size_item == 0)        continue ;
    for (int j = 0; j < size_item; j++) {
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      for(int k=0; k<factors; k++){
        res += u_cu[u*factors+k] * v_cu[i*factors+k];
      }
      prediction_items[index] = res;
    }
    for (int f = 0; f < factors; f++) {
      index_u = u*factors+f;
      ufget = u_cu[index_u];
      for (int j = 0; j < size_item; j++) {
          index = size2+j;
          i = train_spvin_cu[index];
          ifv = v_cu[i*factors+f];
          prediction_items[index] -= ufget * ifv;
          numer[index_u] += (wi_cu[i] - w_cu[index]) * prediction_items[index] * ifv;
        }
        for(int k = 0; k<factors; k++){
          numer[index_u] -= u_cu[u*factors+k] * sv_cu[f*factors+k];
        }
        numer[index_u] += u_cu[index_u] * sv_cu[f*factors+f];
        //printf("%f ", numer[u*factors+f]);
        u_cu[index_u] = numer[index_u] / denom[index_u];
        tmp_uget = u_cu[index_u];
        for (int j = 0; j < size_item; j++){
          i = train_spvin_cu[size2+j];
          prediction_items[size2+j] += tmp_uget * v_cu[i*factors+f];
        }
      }
    }
}

void MF_fastALS::updateUserSchedule(){
  int size, size2;
  float *v_col;
  int total_size = trainMatrix.itemCount()+10;
  //int total_size = 50+10;
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  //save the old U
  float *u_clone, *u_clone_cu;
  cudaHostAlloc((void**)&u_clone, sizeof(float)*factors*userCount, cudaHostAllocDefault);
	#pragma omp parallel for
	for (int i = 0; i<userCount; i++){
    for (int j = 0; j<factors; j++){
      u_clone[i*factors+j] = U.matrix[i][j];
    }
  }
  cudaMalloc((void**)&u_clone_cu, sizeof(float)*userCount*factors);
  cudaMemcpy(u_clone_cu, u_clone, sizeof(float)*userCount*factors, cudaMemcpyHostToDevice);
  float *result, *result_cu;
  cudaHostAlloc((void**)&result, sizeof(float)*(BLOCK_NUM), cudaHostAllocDefault);
  cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));

  float *w_cu, *u_cu, *v_cu, *sv_cu, *train_spvdo_cu;
  int *train_spvin_cu;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&u_cu, sizeof(float)*userCount*factors);
  cudaMalloc((void**)&v_cu, sizeof(float)*itemCount*factors);
  cudaMalloc((void**)&sv_cu, sizeof(float)*factors*factors);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);
  int *train_n, *train_n_cu;
  cudaMalloc((void**)&train_n_cu,sizeof(int)*(userCount+1));
  float *w_h, *u_h, *v_h, *sv_h, *train_spvdo_h;
  int *train_spvin_h;
  cudaHostAlloc((void**)&w_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&u_h, sizeof(float)*userCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&v_h, sizeof(float)*itemCount*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&sv_h, sizeof(float)*factors*factors, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvin_h, sizeof(int)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_spvdo_h, sizeof(float)*total_size, cudaHostAllocDefault);
  cudaHostAlloc((void**)&train_n, sizeof(int)*(userCount+1), cudaHostAllocDefault);
  
  train_n[0] = 0;
  for (int u = 0; u < userCount; u++){
    size = trainMatrix.rows[u].n;;
    size2 = train_n[u];
    train_n[u+1] = size2 + size;

    for(int i=0; i<size; i++){
      w_h[size2+i] = W.rows[u].spv_do[i];
      train_spvin_h[size2+i] = trainMatrix.rows[u].spv_in[i];
      train_spvdo_h[size2+i] = trainMatrix.rows[u].spv_do[i];
    }
  }
  
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      u_h[u*factors+i] = U.matrix[u][i];
  }
  
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      v_h[u*factors+i] = V.matrix[u][i];
  }
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      sv_h[u*factors+i] = SV.matrix[u][i];
  }
  cudaMemcpy(u_cu, u_h, sizeof(float)*userCount*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(v_cu, v_h, sizeof(float)*itemCount*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(sv_cu, sv_h, sizeof(float)*factors*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*(userCount+1), cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);


  float *u_numer, *u_denom;
  cudaMalloc((void**)&u_numer,sizeof(float)*userCount*factors);
  cudaMalloc((void**)&u_denom,sizeof(float)*userCount*factors);

  int dimx, dimy;
  dimx = 64;
  dimy = 16;
  //dim3 block(dimx, dimy);
  //dim3 grid(2);
  dim3 block(THREAD_NUM);
  dim3 grid(dimx, dimy);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  updateUserCuda2<<<grid, block>>>(u_numer, u_denom, v_col, userCount, factors, reg, w_cu, u_cu, v_cu, sv_cu, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout<<"Time of update u: "<<elapsedTime<<" ";

  cudaMemcpy(u_h, u_cu, sizeof(float)*userCount*factors, cudaMemcpyDeviceToHost);


  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++){
      U.matrix[u][i] = u_h[u*factors+i];
      //cout<<U[u][i]<<" ";
    }
  }

  //update su
  for (int f = 0; f < factors; f++) { 
    for (int k = 0; k <= f; k++) {                                          
      float val = SU.matrix[f][k];
      computeSUValue<<<BLOCK_NUM,THREAD_NUM,0>>>(u_clone_cu, u_cu, f, k, factors, result_cu, userCount);
      cudaMemcpy(result, result_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
      #pragma omp parallel for reduction(+:val)                          
      for (int u = 0; u < BLOCK_NUM; u++){                                  
        val += result[u];                                                       
      }                                                                     
      SU.matrix[f][k] = val;
      SU.matrix[k][f] = val;
     }                                                                  
  } 
  //free update su
  cudaFreeHost(u_clone);
  cudaFree(u_clone_cu);
  cudaFreeHost(result);
  cudaFree(result_cu);

  //free update u
  cudaFree(v_col);
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(u_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  cudaFree(wi_cu);
  cudaFree(u_numer);
  cudaFree(u_denom);
  cudaFreeHost(train_n);
  cudaFreeHost(w_h);
  cudaFreeHost(u_h);
  cudaFreeHost(v_h);
  cudaFreeHost(sv_h);
  cudaFreeHost(train_spvin_h);
  cudaFreeHost(train_spvdo_h);

}
*/
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
    float tmp_uget = (*(uget+f));
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
  float wii = Wi[i];
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
    numer *= wii;
    // O(Ni) complexity for the positive ratings part
    float ifget = V.matrix[i][f];
    for (int j=0; j<size_user; j++) {
      u = userList[j];
      ifu = *(u_col+j);
      prediction_users[j] -= ifu * ifget;
      numer += (w_users[j] * rating_users[j] - (w_users[j] - wii) * prediction_users[j]) * ifu;
      denom += (w_users[j] - wii) * ifu * ifu;
    }
    denom += wii * (*(suget+f)) + reg;
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
  delete [] partition_index;
  //cudaFreeHost(u_values);
  //cudaFreeHost(v_values);
}
