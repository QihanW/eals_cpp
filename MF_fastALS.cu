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
#define BLOCK_NUM 64
#define THREAD_NUM 256

/*
using namespace Eigen;

typedef Triplet<float> T_d;
typedef SparseMatrix<float> SpMat;
typedef Matrix<float, Dynamic, 1> VectorXd;
typedef Matrix<float, Dynamic, Dynamic> MatrixXd;
*/

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
	//prediction_users = new float[userCount];
	//prediction_items = new float[itemCount];
	//rating_users = new float[userCount];
	//rating_items = new float[itemCount];
	//w_users = new float[userCount];
	//w_items = new float[itemCount];

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
	//std::vector<T_d> tripletList;
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
/*  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float sum = 0;
  for(int i = bid*THREAD_NUM+tid; i < userCount; i+=BLOCK_NUM*THREAD_NUM){
    sum = sum - ufc[i] * ukc[i] + uf[i] * uk[i];
  }
  result[bid*THREAD_NUM+tid] = sum;*/

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
  int index;
  for(int i = bid*THREAD_NUM+tid; i < userCount; i+=BLOCK_NUM*THREAD_NUM){
    index = i * factors;
    temp = wi_cu[i] * (0 - u_clone[index+f] * u_clone[index+k] + u[index+f] * u[index+k]);
    //temp = temp * wi_cu[i];
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

void MF_fastALS::buildModel() {
	omp_set_num_threads(1);
	
	float loss_pre = FLT_MAX;

  //float su_one[factors * factors];
  //float sv_one[factors * factors];
  /*
  for(int i=0; i<factors; i++){
    for (int j=0; j<factors; j++){
      su_one[i*factors+j] = SU.matrix[i][j];
      sv_one[i*factors+j] = SV.matrix[i][j];
    }
  }
*/


	for (int iter = 0; iter < maxIter; iter++) {
		//std::cout << "Iter: " << iter << " when building model" << std::endl;
		int user_list = 0, item_list = 0;
	  
		DenseMat u_clone(userCount, factors);
		#pragma omp parallel for
		for (int i = 0; i<userCount; i++){
      for (int j = 0; j<factors; j++){
        u_clone.matrix[i][j] = U.matrix[i][j];
      }
    }
 
		double start = omp_get_wtime();
    //#pragma omp parallel for schedule(static, 128) private(prediction_users, rating_users, w_users, V,U)
		#pragma omp parallel for  shared(trainMatrix, W, U, SV, V, Wi)
		for (int u = 0; u < userCount; u++) {
			update_user_thread(u);  
		}
    /*
		#pragma omp parallel for
		for(int f=0; f<64; f++){
		  for(int k=0; k<=f; k++){
        float val = 0;
        for (int u = 0; u < userCount; u++){
           val = val - u_clone.matrix[u][f] * u_clone.matrix[u][k] + U.matrix[u][f] * U.matrix[u][k];
        }
        SU.matrix[f][k] = val;
        SU.matrix[k][f] = val;
      }
    }
    */
   // for (int u = 0; u < userCount; u++){
    //  update_user_SU(u_clone.matrix[u], U.matrix[u]);
   //double end = omp_get_wtime();
   //std::cout<<"First part time: "<<end-start<<std::endl;
  /*  for (int f = 0; f < factors; f++) {                                                                                           
	      for (int k = 0; k <= f; k++) {
	        float val = SU.matrix[f][k];
	        #pragma omp parallel for reduction(+:val) 
	        for (int u = 0; u < userCount; u++){
		          float tmp = 0 - u_clone.matrix[u][f] * u_clone.matrix[u][k] + U.matrix[u][f] * U.matrix[u][k] ;
              val += tmp;
          }
		      SU.matrix[f][k] = val;                                                      
		      SU.matrix[k][f] = val;                                                      
	                                                                               
      }     
    }*/
/*
    int byteSize = sizeof(float)*userCount;
    float *U_clone_f, *U_clone_k, *U_f, *U_k, *result;
    U_clone_f = (float *)malloc(byteSize);
    U_clone_k = (float *)malloc(byteSize);
    U_f = (float *)malloc(byteSize);
    U_k = (float *)malloc(byteSize);
    result = (float *)malloc(sizeof(float)*(BLOCK_NUM));
  */  //std::cout<<"201 is okay"<<std::endl;
    /*cudaHostAlloc((void **)&U_clone_f, byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&U_clone_k, byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&U_f, byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&U_k, byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&result, sizeof(float)*(BLOCK_NUM), cudaHostAllocDefault);

  
    float *U_clone_f_cu, *U_clone_k_cu, *U_f_cu, *U_k_cu, *result_cu;
    cudaMalloc((void**)&U_clone_f_cu, byteSize);
    cudaMalloc((void**)&U_clone_k_cu, byteSize);
    cudaMalloc((void**)&U_f_cu, byteSize);
    cudaMalloc((void**)&U_k_cu, byteSize);
    cudaMalloc((void**)&result_cu, sizeof(float)*(BLOCK_NUM));
    */
    //std::cout<<"208 is okay"<<std::endl;
  
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
        /*
        #pragma omp parallel for
        for (int u = 0; u < userCount; u++){
          U_clone_f[u] = u_clone.matrix[u][f];
          U_clone_k[u] = u_clone.matrix[u][k];
          U_f[u] = U.matrix[u][f];
          U_k[u] = U.matrix[u][k];
         }
        */
       // #pragma omp parallel for
       // for (int u = 0; u < BLOCK_NUM; u++)
     //     result[u] = 0;
        /*
        cudaMemcpy(U_clone_f_cu, U_clone_f, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(U_clone_k_cu, U_clone_k, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(U_f_cu, U_f, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(U_k_cu, U_k, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(U_k_cu, U_k, byteSize, cudaMemcpyHostToDevice);
        */
        //cudaMemcpy(result_cu, result, sizeof(float)*(BLOCK_NUM), cudaMemcpyHostToDevice);

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

    //std::cout<<"245 is okay"<<std::endl;
    cudaFree(U_clone_cu);
    //cudaFree(U_clone_k_cu);
    cudaFree(U_cu);
    //cudaFree(U_k_cu);
    cudaFree(result_cu);
    /*free(U_clone_f);
    free(U_clone_k);
    free(U_f);
    free(U_k);
    free(result);*/
    cudaFreeHost(U_clone_h);
    //cudaFreeHost(U_clone_k);
    cudaFreeHost(U_h);
    //cudaFreeHost(U_k);
    cudaFreeHost(result);

   // std::cout<<"The second part: "<<omp_get_wtime() - end<<std::endl; 
    double time_user_update = omp_get_wtime() - start;
		std::cout << "Time of user_update: " <<time_user_update<< std::endl;
    //std::cout << "User list size: "<<user_list;
		// Update item latent vectors
		//
		//
		//
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
     // std::cout << i << std::endl;
		}
		
		/*for (int i = 0; i < itemCount; i++){
      update_item_SV(i, v_clone.matrix[i], V.matrix[i]);
    }*/
/*  
  for (int f = 0; f < factors; f++) {
	  for (int k = 0; k <= f; k++) {
		  float val = SV.matrix[f][k];
 		  
      #pragma omp parallel for reduction(+:val)
  		for (int u = 0; u < itemCount; u++){
 			  float tmp = 0 - v_clone.matrix[u][f] * v_clone.matrix[u][k] + V.matrix[u][f] * V.matrix[u][k] ;
			  tmp = tmp *  Wi[u];
			  val += tmp;
		 }
      SV.matrix[f][k] = val;
      SV.matrix[k][f] = val;
    }
  }
  */

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
  //std::cout << "165: " <<L <<endl;
	int i;
	for (int u = 0; u < userCount; u++) {
		float l = 0;
		//std::vector<int> itemList;
		//itemList = trainMatrix.getRowRef(u).indexList();
		int *itemList;
		itemList = trainMatrix.rows[u].spv_in;
		int size_item = trainMatrix.rows[u].n;
		for (int j = 0; j < size_item; j++) {
			i = itemList[j];
			float pred = predict(u, i);
			l += W.rows[u].spv_do[j] * pow(trainMatrix.rows[u].spv_do[j] - pred, 2);
			l -= Wi[i] * pow(pred, 2);
		}
		//if (u<10)
    //  std::cout<<l<<endl;
		//MatrixXd u_temp = U.getData();
		//MatrixXd sv_temp = SV.getData();
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
		//int size_avx = size_item / NUM_float;
		//int remain = size_item % NUM_float;
		
   // int res = itemList.size();
    //clock_t start = clock();
    if (size_item == 0)        return ;    // user has no ratings
    int i;
    
    //float pp[size_item];
    //float rr[size_item];
    //float ww[size_item];
    
    //float test[size_avx];

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
     // float *numer_tmp = new float[NUM_float];
      //float *vget = V.matrix[u];
    for (int f = 0; f < factors; f++) {
      float numer = 0, denom = 0;
      // O(K) complexity for the negative part
//      #pragma omp parallel num_thread(16)
  //    {
    //  #pragma omp for reduction(-:numer)
      //float *uget = U.matrix[u];
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
      float mius = -1;
      

      for (int j = 0; j<size_item; j++) {
        i = itemList[j];
        ifv = *(v_col+j);
        prediction_items[j] -= ufget * ifv;
        numer += (w_items[j] * rating_items[j] - (w_items[j] - Wi[i]) * prediction_items[j]) * ifv;
        denom += (w_items[j] - Wi[i]) * ifv * ifv;
      }
     // }
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
    //delete [] numer_tmp;
    //delete [] itemL;
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

    //float test[size_avx];

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
    //float *numer_tmp = new float[NUM_float];

    //DenseVec oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
      // O(K) complexity for the w0 part
      float numer = 0, denom = 0;
      //float *vget = V.matrix[i];
      float *suget = SU.matrix[f];
      
      for(int j = 0; j<size_user; j++){
        u = userList[j];
        u_col[j] = U.matrix[u][f];
      }
      
      /*
      _mm512_store_pd(numer_tmp, _mm512_setzero_pd ());

      for (int k = 0; k < factors; k+=NUM_float) {
        //if (k != f)
          //numer -= V.get(i, k) * SU.get(f, k);
          //numer -= (*(vget+k)) * (*(suget+k));
        __m512d uget_k = _mm512_load_pd(vget + k);
        __m512d svget_k = _mm512_load_pd(suget + k);
        __m512d num = _mm512_load_pd(numer_tmp);
        __m512d tmp_mul = _mm512_mul_pd(uget_k, svget_k);
        __m512d tmp_add = _mm512_add_pd(num, tmp_mul);
        _mm512_store_pd(numer_tmp, tmp_add);
      }
         
      for(int k=0; k<NUM_float; k++){
          numer -= numer_tmp[k];
      }

      numer += (*(vget+f)) * (*(suget+f));
      */
      
      for (int k = 0; k < factors; k++) {
        if (k != f)
          //numer -= V.get(i, k) * SU.get(f, k);
          numer -= (*(vget+k)) * (*(suget+k));
      }
      
      numer *= Wi[i];

     // std::cout << "Time of 312: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // O(Ni) complexity for the positive ratings part
      float ifget = V.matrix[i][f];
      float mius = -1;
      
      /*
      for (int j=0; j<size_avx; j+=NUM_float){
        __m512d uget_avx = _mm512_set1_pd(ifget); 
        __m512d uget_avx2 = _mm512_set1_pd(mius); 
        __m512d vcol_avx = _mm512_load_pd(u_col + j);
        __m512d tmp_mul = _mm512_mul_pd(uget_avx, vcol_avx);
        __m512d tmp_mul2 = _mm512_mul_pd(uget_avx2, tmp_mul);
        __m512d pre_avx = _mm512_load_pd(prediction_users + j);
        __m512d sum_avx = _mm512_add_pd(tmp_mul2, pre_avx);
        _mm512_store_pd(test + j, sum_avx);
      }
      
      for(int j=0; j<size_avx; j++)
          prediction_users[j] = test[j]; 
    
      for(int j=size_avx; j<size_user; j++){
          ifu = *(u_col+j);
          prediction_users[j] -= ifget * ifu;
      }
      */

      for (int j=0; j<size_user; j++) {
        u = userList[j];
        ifu = *(u_col+j);
        prediction_users[j] -= ifu * ifget;
        numer += (w_users[j] * rating_users[j] - (w_users[j] - Wi[i]) * prediction_users[j]) * ifu;
        denom += (w_users[j] - Wi[i]) * ifu * ifu;
      }
      denom += Wi[i] * (*(suget+f)) + reg;

     //std::cout << "Time of 322: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // Parameter update
     // V.set(i, f, numer / denom);
      // Update the prediction cache for the item
      (*(vget+f)) = numer / denom;
      float tmp_vget = numer / denom;
      /*
      for (int j=0; j<size_avx; j+=NUM_float){
        __m512d uget_avx = _mm512_set1_pd(tmp_vget);
        __m512d vcol_avx = _mm512_load_pd(u_col + j);
        __m512d tmp_mul = _mm512_mul_pd(uget_avx, vcol_avx);
        __m512d pre_avx = _mm512_load_pd(prediction_users + j);
        __m512d sum_avx = _mm512_add_pd(tmp_mul, pre_avx);
        _mm512_store_pd(test + j, sum_avx);
      
      }

      for (int j=0; j<size_avx; j++)
         prediction_users[j] = test[j];
      for (int j = size_avx; j<size_user; j++){
         prediction_users[j] += tmp_vget * u_col[j];
      }
      */
      for(int j=0; j<size_user; j++)
        prediction_users[j] += tmp_vget * u_col[j];
      //for (int u : userList)

        //  prediction_users[u] +=  U.matrix[u][f] * (*(vget+f)) ;
    } // end for f

   delete [] prediction_users;
   delete [] rating_users;
   delete [] w_users;
    delete [] u_col;
    //delete [] numer_tmp;
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


/*
void MF_fastALS::update_user(int u) {
   //omp_set_num_threads(16);
    std::vector<int> itemList;
		itemList = trainMatrix.getRowRef(u).indexList();
   // int res = itemList.size();
    //clock_t start = clock();
    if (itemList.size() == 0)        return ;    // user has no ratings
    // prediction cache for the user
    //std::cout << "210" << std::endl;
  //  std::cout << "Time of 213: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    //std::cout << itemList.size() << std::endl;
    //#pragma omp parallel for 
    for (int i : itemList) {
      //start = clock();
      prediction_items[i] = predict(u, i);
    //start = clock();
      //std::cout << "Time of 218: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_items[i] = trainMatrix.getValue(u, i);
      //std::cout << "Time of 220: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      w_items[i] = W.getValue(u, i);
      //std::cout << "Time of 222: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //std::cout << "217" << std::endl;
    //std::cout << "Time of 245: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    DenseVec oldVector = U.row(u);
      float *uget = U.matrix[u];
    for (int f = 0; f < factors; f++) {
      float numer = 0, denom = 0;
      // O(K) complexity for the negative part
//      #pragma omp parallel num_thread(16)
  //    {
    //  #pragma omp for reduction(-:numer)
      //float *uget = U.matrix[u];
      float *svget = SV.matrix[f];
      for (int k = 0; k < factors; k++) {
        if (k != f)
          numer -= (*(uget+k)) * (*(svget+k));
      }
     // }
      //numer *= w0;
      // O(Nu) complexity for the positive part
      //clock_t start = clock();
     // float numer2 = 0;
      //#pragma omp parallel num_thread(16)                                                      
     // {
     // #pragma omp for reduction(+:numer2) reduction(+:denom)
      float ufget = U.matrix[u][f];
      for (int i : itemList) {
        float ifv = V.matrix[i][f];
        prediction_items[i] -= ufget * ifv;
        numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * ifv;
        //int x =  (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]);
        //numer += x * V.get(i,f);
        
         denom += (w_items[i] - Wi[i]) * ifv * ifv;
      }
     // }
      denom += SV.get(f, f) + reg;
      
      // Parameter Update
      U.set(u, f, numer / denom);

      // Update the prediction cache
//      #pragma omp parallel for shared(prediction_items)  
      for (int i : itemList)
        prediction_items[i] += U.get(u, f) * V.get(i, f);
    } // end for f
    //std::cout << "242" << std::endl;
   // std::cout << "Time of 283: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    // Update the SU cache
    //start = clock();
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        float val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k) + (*(uget+f)) * (*(uget+k));
        SU.set(f, k, val);
        SU.set(k, f, val);
      }
    //std::cout << "Time of 270: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //return res;
    //std::cout << "Time of 294: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
  }

void  MF_fastALS::update_item(int i) {
    std::vector<int> userList;
    //clock_t	start = clock();
  //  std::cout << "Time of 281: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    userList = trainMatrix.getColRef(i).indexList();
    //int res = userList.size();
    if (userList.size() == 0)        return; // item has no ratings.
    // prediction cache for the item
   //  std::cout << "Time of 286: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    for (int u : userList) {
      //if(u<25677){

    //	   if (i == 25458)
      //   std::cout << u << "Time of 288: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      prediction_users[u] = predict(u, i);
      //if (i == 25458)
        // std::cout << u << " Time of 292: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_users[u] = trainMatrix.getValue(u, i);
      w_users[u] = W.getValue(u, i);
     // }
    }

    //std::cout << "Time of 300: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    float *vget = V.matrix[i];

    DenseVec oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
      // O(K) complexity for the w0 part
      float numer = 0, denom = 0;
      //float *vget = V.matrix[i];
      float *suget = SU.matrix[f];

      for (int k = 0; k < factors; k++) {
        if (k != f)
          //numer -= V.get(i, k) * SU.get(f, k);
          numer -= (*(vget+k)) * (*(suget+k));
      }
      numer *= Wi[i];

     // std::cout << "Time of 312: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // O(Ni) complexity for the positive ratings part
      float ifv = V.matrix[i][f];
      for (int u : userList) {
        float ufu = U.matrix[u][f];
        prediction_users[u] -= ufu * ifv;
        numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * ufu;
        denom += (w_users[u] - Wi[i]) * ufu * ufu;
      }
      denom += Wi[i] * SU.get(f, f) + reg;

     //std::cout << "Time of 322: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // Parameter update
      V.set(i, f, numer / denom);
      // Update the prediction cache for the item
      for (int u : userList)

          prediction_users[u] += U.get(u, f) * V.get(i, f);
    } // end for f

   // std::cout << "Time of 331: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    // Update the SV cache
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        float val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * Wi[i]
          + (*(vget+f)) * (*(vget+k)) * Wi[i];
        SV.set(f, k, val);
        SV.set(k, f, val);
      }
    }
    //std::cout << "Time of 341: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
 
   //return res;
}
*/
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
    //td::cout<<"389"<<endl;
    //clock_t start = clock(); 
    std::vector<float> result(3);
    std::map<int, float> map_item_score;
    float maxScore;
    //    int gtItem = testRatings[u].itemId;
    //        float maxScore = predict(u, gtItem)
    maxScore = predict(u, gtItem);
    int countLarger = 0;
    for (int i = 0; i < itemCount; i++) {
      float score = predict(u, i);
      map_item_score.insert(std::make_pair(i, score));
      if (score > maxScore) countLarger++;
      if (countLarger > topK)  return result;
      //            if (countLarger > topK){
      //                hits[u]  = result[0];
      //                ndcgs[u] = result[1];
      //                precs[u] = result[2];
    }
   // std::cout << "Time of 408: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    std::vector<int> rankList;
    std::vector<std::pair<int, float>>top_K(topK);
    std::partial_sort_copy(map_item_score.begin(),
                 map_item_score.end(),
                         top_K.begin(),
                         top_K.end(),
                         [](std::pair<const int, int> const& l,
                       std::pair<const int, int> const& r)
                 {
                return l.second > r.second;
                  });
    for (auto const& p : top_K)
    {
      rankList.push_back(p.first);

    }
   // std::cout << "Time of 425: " <<(float)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
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
  
     //int *itemList;
     //itemList = trainMatrix.rows[u].spv_in;
     //int size_item = trainMatrix.rows[u].n;
     //for (int j = 0; j < size_item; j++) {
     i = testRatings[u].itemId;
     float score = testRatings[u].score;
     float pred = predict(u, i);
     l += W.getValue(u, i) * pow(score - pred, 2);
     l -= Wi[i] * pow(pred, 2);

     l += SV.mult(U.row_fal(u)).inner(U.row_fal(u));
     L += l;
       //std::cout<<L<<endl;
   }
  

  
  
  
  
//  float res = loss();
 /* for(int u=0; u<userCount; u++){
    int item_id = testRatings[u].itemId;
    float score = testRatings[u].score;
    float pre = predict(u, item_id);
    res += (pre-score) * (pre-score);
  }
  res = sqrt(res / userCount);
  */
  return L / userCount ;
}

  MF_fastALS::~MF_fastALS()
  {
    //delete [] prediction_users;
    //delete [] prediction_items;
    //delete [] rating_users;
    //delete [] rating_items;
    //delete [] w_users;
    //delete [] w_items;
    delete [] Wi;
}
