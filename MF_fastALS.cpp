#include "omp.h"
#include"MF_fastALS.h"
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
#include "DenseVec.h"
#include "DenseMat.h"
#include "SparseVec.h"
#include "SparseMat.h"
#include "Rating.h"

/*
using namespace Eigen;

typedef Triplet<double> T_d;
typedef SparseMatrix<double> SpMat;
typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
*/

MF_fastALS::MF_fastALS(SparseMat trainMatrix1, std::vector<Rating> testRatings1,
	int topK1, int threadNum1, int factors1, int maxIter1, double w01, double alpha1, double reg1,
	double init_mean1, double init_stdev1, bool showProgress1, bool showLoss1, int userCount1,
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
	prediction_users = new double[userCount];
	prediction_items = new double[itemCount];
	rating_users = new double[userCount];
	rating_items = new double[itemCount];
	w_users = new double[userCount];
	w_items = new double[itemCount];

	// Set the Wi as a decay function w0 * pi ^ alpha
	double sum = 0, Z = 0;
	double *p = new double[itemCount];
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
	Wi = new double[itemCount];
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

void MF_fastALS::buildModel() {
	//omp_set_num_threads(256);
	double loss_pre = DBL_MAX;
	for (int iter = 0; iter < maxIter; iter++) {
		//std::cout << "Iter: " << iter << " when building model" << std::endl;
		int user_list = 0, item_list = 0;
		
		DenseMat u_clone(userCount, factors);
		for (int i = 0; i<userCount; i++){
      for (int j = 0; j<factors; j++){
        u_clone.matrix[i][j] = U.matrix[i][j];
      }
    }
		clock_t start = clock();
    //#pragma omp parallel for schedule(static, 128) private(prediction_users, rating_users, w_users, V,U)
		for (int u = 0; u < userCount; u++) {
			update_user_thread(u);  
		}
    for (int u = 0; u < userCount; u++){
      update_user_SU(u_clone.matrix[u], U.matrix[u]);
    }
    double time_user_update = (double)(clock() - start)/CLOCKS_PER_SEC;
		std::cout << "Time of user_update: " <<time_user_update<< std::endl;
    //std::cout << "User list size: "<<user_list;
		// Update item latent vectors
	 DenseMat v_clone(itemCount, factors);
    for (int i = 0; i<itemCount; i++){
	    for (int j = 0; j<factors; j++){
		    v_clone.matrix[i][j] = V.matrix[i][j];
	     }
     }
	  
	  
	  start = clock();
		for (int i = 0; i < itemCount; i++) {
      update_item_thread(i);
     // std::cout << i << std::endl;
		}
		for (int i = 0; i < itemCount; i++){
      update_item_SV(i, v_clone.matrix[i], V.matrix[i]);
    }
      
    double time_item_update = (double)(clock() - start)/CLOCKS_PER_SEC;
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

double MF_fastALS::showLoss(int iter, double time, double loss_pre) {
	clock_t end = clock();
	double loss_cur = loss();
	std::string symbol = loss_pre >= loss_cur ? "-" : "+";
	std::cout << "Iter=" << iter << " " <<time << " " << symbol << " loss:" << loss_cur << " " <<(double)(clock() - end)/ CLOCKS_PER_SEC << std::endl;

	return loss_cur;
}

double MF_fastALS::loss() {
	double L = reg * (U.squaredSum() + V.squaredSum());
  //std::cout << "165: " <<L <<endl;
	for (int u = 0; u < userCount; u++) {
		double l = 0;
		std::vector<int> itemList;
		itemList = trainMatrix.getRowRef(u).indexList();
		for (int i : itemList) {
			double pred = predict(u, i);
			l += W.getValue(u, i) * pow(trainMatrix.getValue(u, i) - pred, 2);
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

double MF_fastALS::predict(int u, int i) {

 //double res =  u_temp.row(u) * v_temp.row(i).transpose();
 //double res =  U.row_fal(u).inner(V.row_fal(i));
 double * u_tmp = U.matrix[u];
 double * v_tmp = V.matrix[i];
 double res = 0;
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
          double val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * Wi[i];
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
    
    std::vector<int> itemList;
		itemList = trainMatrix.getRowRef(u).indexList();
   // int res = itemList.size();
    //clock_t start = clock();
    if (itemList.size() == 0)        return ;    // user has no ratings
    
    //double *prediction_items = new double[itemCount];
    //double *rating_items = new double[itemCount]; 
   // double *w_items = new double[itemCount]; 

    // prediction cache for the user
    

  //  std::cout << "Time of 213: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    //std::cout << itemList.size() << std::endl;
    //#pragma omp parallel for 
    for (int i : itemList) {
      //start = clock();
      prediction_items[i] = predict(u, i);
    //start = clock();
      //std::cout << "Time of 218: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_items[i] = trainMatrix.rows[u].getValue(i);
      //std::cout << "Time of 220: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      w_items[i] = W.getValue(u, i);
      //std::cout << "Time of 222: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //std::cout << "217" << std::endl;
    //std::cout << "Time of 245: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    //DenseVec oldVector = U.row(u);
      double *uget = U.matrix[u];
      //double *vget = V.matrix[u];
    for (int f = 0; f < factors; f++) {
      double numer = 0, denom = 0;
      // O(K) complexity for the negative part
//      #pragma omp parallel num_thread(16)
  //    {
    //  #pragma omp for reduction(-:numer)
      //double *uget = U.matrix[u];
      double *svget = SV.matrix[f];
      for (int k = 0; k < factors; k++) {
        if (k != f)
          numer -= (*(uget+k)) * (*(svget+k));
      }
     // }
      //numer *= w0;
      // O(Nu) complexity for the positive part
      //clock_t start = clock();
     // double numer2 = 0;
      //#pragma omp parallel num_thread(16)                                                      
     // {
     // #pragma omp for reduction(+:numer2) reduction(+:denom)
      double ufget = U.matrix[u][f];
      for (int i : itemList) {
        double ifv = V.matrix[i][f];
        prediction_items[i] -= ufget * ifv;
        numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * ifv;
        //int x =  (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]);
        //numer += x * V.get(i,f);
        
         denom += (w_items[i] - Wi[i]) * ifv * ifv;
      }
     // }
      denom +=(*(svget+f)) + reg;
      
      // Parameter Update
      (*(uget+f)) = numer / denom;

      // Update the prediction cache
//      #pragma omp parallel for shared(prediction_items)  
      for (int i : itemList)
        prediction_items[i] +=  (*(uget+f)) * V.matrix[i][f];
    } // end for f

    //delete [] prediction_items;
    //delete [] rating_items;
    //delete [] w_items;

}

void MF_fastALS::update_user_SU(double *oldVector, double *uget){
  double ** suget = SU.matrix;
  double * sugetu;
  for (int f = 0; f < factors; f++) {
      sugetu = *( suget + f);
      for (int k = 0; k <= f; k++) {
        double val = (*(sugetu + k)) - (*(oldVector+f)) * (*(oldVector+k)) + (*(uget+f)) * (*(uget+k));
        SU.set(f, k, val);
        SU.set(k, f, val);
      }
    }
}


void  MF_fastALS::update_item_thread(int i){
  std::vector<int> userList;
    //clock_t	start = clock();
  //  std::cout << "Time of 281: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    userList = trainMatrix.getColRef(i).indexList();
    //int res = userList.size();
    if (userList.size() == 0)        return; // item has no ratings.
    // prediction cache for the item
   //  std::cout << "Time of 286: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    //double *prediction_users = new double[userCount];
    //double *rating_users = new double[userCount];
    //double *w_users = new double[userCount];

    
    for (int u : userList) {
      //if(u<25677){

    //	   if (i == 25458)
      //   std::cout << u << "Time of 288: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      prediction_users[u] = predict(u, i);
      //if (i == 25458)
        // std::cout << u << " Time of 292: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_users[u] = trainMatrix.getValue(u, i);
      w_users[u] = W.getValue(u, i);
     // }
    }

    //std::cout << "Time of 300: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    double *vget = V.matrix[i];

    //DenseVec oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
      // O(K) complexity for the w0 part
      double numer = 0, denom = 0;
      //double *vget = V.matrix[i];
      double *suget = SU.matrix[f];

      for (int k = 0; k < factors; k++) {
        if (k != f)
          //numer -= V.get(i, k) * SU.get(f, k);
          numer -= (*(vget+k)) * (*(suget+k));
      }
      numer *= Wi[i];

     // std::cout << "Time of 312: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // O(Ni) complexity for the positive ratings part
      double ifv = V.matrix[i][f];
      for (int u : userList) {
        double ufu = U.matrix[u][f];
        prediction_users[u] -= ufu * ifv;
        numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * ufu;
        denom += (w_users[u] - Wi[i]) * ufu * ufu;
      }
      denom += Wi[i] * (*(suget+f)) + reg;

     //std::cout << "Time of 322: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // Parameter update
     // V.set(i, f, numer / denom);
      // Update the prediction cache for the item
      (*(vget+f)) = numer / denom;
      for (int u : userList)

          prediction_users[u] +=  U.matrix[u][f] * (*(vget+f)) ;
    } // end for f

   //delete [] prediction_users;
   //delete [] rating_users;
   //delete [] w_users;

}

void MF_fastALS::update_item_SV(int i, double *oldVector, double *vget){
  double ** svget = SV.matrix;
  double * svgeti;
  
  for (int f = 0; f < factors; f++) {
    svgeti = *(svget + f);  
    for (int k = 0; k <= f; k++) {
        double val = (*(svgeti + k)) - (*(oldVector+f)) * (*(oldVector+k)) * Wi[i]
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
  //  std::cout << "Time of 213: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    //std::cout << itemList.size() << std::endl;
    //#pragma omp parallel for 
    for (int i : itemList) {
      //start = clock();
      prediction_items[i] = predict(u, i);
    //start = clock();
      //std::cout << "Time of 218: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_items[i] = trainMatrix.getValue(u, i);
      //std::cout << "Time of 220: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      w_items[i] = W.getValue(u, i);
      //std::cout << "Time of 222: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //std::cout << "217" << std::endl;
    //std::cout << "Time of 245: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    DenseVec oldVector = U.row(u);
      double *uget = U.matrix[u];
    for (int f = 0; f < factors; f++) {
      double numer = 0, denom = 0;
      // O(K) complexity for the negative part
//      #pragma omp parallel num_thread(16)
  //    {
    //  #pragma omp for reduction(-:numer)
      //double *uget = U.matrix[u];
      double *svget = SV.matrix[f];
      for (int k = 0; k < factors; k++) {
        if (k != f)
          numer -= (*(uget+k)) * (*(svget+k));
      }
     // }
      //numer *= w0;
      // O(Nu) complexity for the positive part
      //clock_t start = clock();
     // double numer2 = 0;
      //#pragma omp parallel num_thread(16)                                                      
     // {
     // #pragma omp for reduction(+:numer2) reduction(+:denom)
      double ufget = U.matrix[u][f];
      for (int i : itemList) {
        double ifv = V.matrix[i][f];
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
   // std::cout << "Time of 283: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    // Update the SU cache
    //start = clock();
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k) + (*(uget+f)) * (*(uget+k));
        SU.set(f, k, val);
        SU.set(k, f, val);
      }
    //std::cout << "Time of 270: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //return res;
    //std::cout << "Time of 294: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
  }

void  MF_fastALS::update_item(int i) {
    std::vector<int> userList;
    //clock_t	start = clock();
  //  std::cout << "Time of 281: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    userList = trainMatrix.getColRef(i).indexList();
    //int res = userList.size();
    if (userList.size() == 0)        return; // item has no ratings.
    // prediction cache for the item
   //  std::cout << "Time of 286: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    for (int u : userList) {
      //if(u<25677){

    //	   if (i == 25458)
      //   std::cout << u << "Time of 288: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      prediction_users[u] = predict(u, i);
      //if (i == 25458)
        // std::cout << u << " Time of 292: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      rating_users[u] = trainMatrix.getValue(u, i);
      w_users[u] = W.getValue(u, i);
     // }
    }

    //std::cout << "Time of 300: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    double *vget = V.matrix[i];

    DenseVec oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
      // O(K) complexity for the w0 part
      double numer = 0, denom = 0;
      //double *vget = V.matrix[i];
      double *suget = SU.matrix[f];

      for (int k = 0; k < factors; k++) {
        if (k != f)
          //numer -= V.get(i, k) * SU.get(f, k);
          numer -= (*(vget+k)) * (*(suget+k));
      }
      numer *= Wi[i];

     // std::cout << "Time of 312: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // O(Ni) complexity for the positive ratings part
      double ifv = V.matrix[i][f];
      for (int u : userList) {
        double ufu = U.matrix[u][f];
        prediction_users[u] -= ufu * ifv;
        numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * ufu;
        denom += (w_users[u] - Wi[i]) * ufu * ufu;
      }
      denom += Wi[i] * SU.get(f, f) + reg;

     //std::cout << "Time of 322: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // Parameter update
      V.set(i, f, numer / denom);
      // Update the prediction cache for the item
      for (int u : userList)

          prediction_users[u] += U.get(u, f) * V.get(i, f);
    } // end for f

   // std::cout << "Time of 331: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    // Update the SV cache
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * Wi[i]
          + (*(vget+f)) * (*(vget+k)) * Wi[i];
        SV.set(f, k, val);
        SV.set(k, f, val);
      }
    }
    //std::cout << "Time of 341: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
 
   //return res;
}
*/
  void MF_fastALS::initS() {
    SU = U.transpose().mult(U);
    SV= DenseMat(factors, factors);
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        double val = 0;
        for (int i = 0; i < itemCount; i++)
          val += V.get(i, f) * V.get(i, k) * Wi[i];
        SV.set(f, k, val);
        SV.set(k, f, val);
      }
    }
  }

  double MF_fastALS::getHitRatio(std::vector<int> rankList, int gtItem) {
    for (int item : rankList) {
      if (item == gtItem)    return 1;
    }
    return 0;
  }
  double MF_fastALS::getNDCG(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
      int item = rankList[i];
      if (item == gtItem)
        return log(2) / log(i + 2);
    }
    return 0;
  }
  double MF_fastALS::getPrecision(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
      int item = rankList[i];
      if (item == gtItem)
        return 1.0 / (i + 1);
    }
    return 0;
  }

  std::vector<double> MF_fastALS::evaluate_for_user(int u, int gtItem, int topK) {
    //td::cout<<"389"<<endl;
    //clock_t start = clock(); 
    std::vector<double> result(3);
    std::map<int, double> map_item_score;
    double maxScore;
    //    int gtItem = testRatings[u].itemId;
    //        double maxScore = predict(u, gtItem)
    maxScore = predict(u, gtItem);
    int countLarger = 0;
    for (int i = 0; i < itemCount; i++) {
      double score = predict(u, i);
      map_item_score.insert(std::make_pair(i, score));
      if (score > maxScore) countLarger++;
      if (countLarger > topK)  return result;
      //            if (countLarger > topK){
      //                hits[u]  = result[0];
      //                ndcgs[u] = result[1];
      //                precs[u] = result[2];
    }
   // std::cout << "Time of 408: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    std::vector<int> rankList;
    std::vector<std::pair<int, double>>top_K(topK);
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
   // std::cout << "Time of 425: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    result[0] = getHitRatio(rankList, gtItem);
    result[1] = getNDCG(rankList, gtItem);
    result[2] = getPrecision(rankList, gtItem);
    return result;
  }

  MF_fastALS::~MF_fastALS()
  {
    delete [] prediction_users;
    delete [] prediction_items;
    delete [] rating_users;
    delete [] rating_items;
    delete [] w_users;
    delete [] w_items;
    delete [] Wi;
}
