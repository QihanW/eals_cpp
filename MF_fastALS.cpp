//#include "omp.h"
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
	prediction_users.resize(userCount);
	prediction_items.resize(itemCount);
	rating_users.resize(userCount);
	rating_items.resize(itemCount);
	w_users.resize(userCount);
	w_items.resize(itemCount);

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
	Wi.resize(itemCount);
	for (int i = 0; i < itemCount; i++)
		Wi[i] = w0 * p[i] / Z;

	// By default, the weight for positive instance is uniformly 1
	W = SparseMat(userCount, itemCount);
	//std::vector<T_d> tripletList;
	for (int u = 0; u < userCount; u++){
    std::vector<int> tmp = trainMatrix.getRowRef(u).indexList();
    vector<int>::iterator iter;
    int i = 0;
    for (iter = tmp.begin(); iter != tmp.end(); iter++){
      W.setValue(u, i, 1);
      i++;
    }
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
    std::vector<int> tmp = trainMatrix.getRowRef(u).indexList();
    vector<int>::iterator iter;
    int i = 0;
    for (iter = tmp.begin(); iter != tmp.end(); iter++){
      W.setValue(u, i, 1);
      i++;
    }
  }
}

void MF_fastALS::setUV(DenseMat U, DenseMat V) {
	this->U = U.clone();
	this->V = V.clone();
	initS();
}

void MF_fastALS::buildModel() {
	double loss_pre = DBL_MAX;
	for (int iter = 0; iter < maxIter; iter++) {
		//std::cout << "Iter: " << iter << " when building model" << std::endl;
		clock_t start = clock();
		for (int u = 0; u < userCount; u++) {
			update_user(u);

		}
		std::cout << "Time of user_update: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
		// Update item latent vectors
		for (int i = 0; i < itemCount; i++) {
    	update_item(i);
     // std::cout << i << std::endl;
		}
    //std::cout << "end of item" << std::endl;
    std::cout << "Time of item_update: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
		// Show loss
		if (showloss)
			loss_pre = showLoss(iter, start, loss_pre);

	} // end for iter
}

void MF_fastALS::runOneIteration() {
	// Update user latent vectors
	for (int u = 0; u < this->userCount; u++) {
		update_user(u);
	}

	// Update item latent vectors
	for (int i = 0; i < this->itemCount; i++) {
		update_item(i);
	}
}

double MF_fastALS::showLoss(int iter, long start, double loss_pre) {
	clock_t end = clock();
	double loss_cur = loss();
	std::string symbol = loss_pre >= loss_cur ? "-" : "+";
	std::cout << "Iter=" << iter << " " <<(double)(end - start)/CLOCKS_PER_SEC << " " << symbol << " loss:" << loss_cur << " " <<(double)(clock() - end)/ CLOCKS_PER_SEC << std::endl;

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
	
//	clock_t start = clock();
	//MatrixXd u_temp = U.getData();
  //std::cout << "Time of 181: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
	//MatrixXd v_temp = V.getData();
 // std::cout << "Time of 183: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
 //double res =  u_temp.row(u) * v_temp.row(i).transpose();
 double res =  U.row_fal(u).inner(V.row_fal(i));
 //std::cout << "Time of 185: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
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
      update_user(u);
      update_item(i);
    }
  }

  void MF_fastALS::update_user(int u) {
   //omp_set_num_threads(16);
    std::vector<int> itemList;
		itemList = trainMatrix.getRowRef(u).indexList();
    
    //clock_t start = clock();
    
    if (itemList.size() == 0)        return;    // user has no ratings
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
    //std::cout << "Time of 234: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    DenseVec oldVector = U.row(u);
    //start = clock();
    for (int f = 0; f < factors; f++) {
      double numer = 0, denom = 0;
      // O(K) complexity for the negative part
//      #pragma omp parallel num_thread(16)
  //    {
    //  #pragma omp for reduction(-:numer)
      for (int k = 0; k < factors; k++) {
        if (k != f)
          numer -= U.get(u, k) * SV.get(f, k);
      }
     // }
      //numer *= w0;
      // O(Nu) complexity for the positive part
      //clock_t start = clock();
      double numer2 = 0;
      //#pragma omp parallel num_thread(16)                                                      
     // {
     // #pragma omp for reduction(+:numer2) reduction(+:denom)
      for (int i : itemList) {
        prediction_items[i] -= U.get(u, f) * V.get(i, f);
        numer2 += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * V.get(i, f);
        denom += (w_items[i] - Wi[i]) * V.get(i, f) * V.get(i, f);
      }
     // }
      numer += numer2;
      denom += SV.get(f, f) + reg;
      
      // Parameter Update
      U.set(u, f, numer / denom);

      // Update the prediction cache
//      #pragma omp parallel for shared(prediction_items)  
      for (int i : itemList)
        prediction_items[i] += U.get(u, f) * V.get(i, f);
    } // end for f
    //std::cout << "242" << std::endl;
    //std::cout << "Time of 261: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    // Update the SU cache
    //start = clock();
    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k) + U.get(u, f) * U.get(u, k);
        SU.set(f, k, val);
        SU.set(k, f, val);
      }
    //std::cout << "Time of 270: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    }
    //std::cout << "Time of 271: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
  }

  void MF_fastALS::update_item(int i) {
    std::vector<int> userList;
    clock_t	start = clock();
  //  std::cout << "Time of 281: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
    userList = trainMatrix.getColRef(i).indexList();
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

    DenseVec oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
      // O(K) complexity for the w0 part
      double numer = 0, denom = 0;
      for (int k = 0; k < factors; k++) {
        if (k != f)
          numer -= V.get(i, k) * SU.get(f, k);
      }
      numer *= Wi[i];

     // std::cout << "Time of 312: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
      // O(Ni) complexity for the positive ratings part
      for (int u : userList) {
        
        prediction_users[u] -= U.get(u, f) * V.get(i, f);
        numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U.get(u, f);
        denom += (w_users[u] - Wi[i]) * U.get(u, f) * U.get(u, f);
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
          + V.get(i, f) * V.get(i, k) * Wi[i];
        SV.set(f, k, val);
        SV.set(k, f, val);
      }
    }
    //std::cout << "Time of 341: " <<(double)(clock() - start)/CLOCKS_PER_SEC  << std::endl;
  }

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
}
