#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

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

using namespace Eigen;

typedef Triplet<double> T_d;

MF_fastALS::MF_fastALS(SparseMat trainMatrix, std::vector<Rating> testRatings,
	int topK, int threadNum, int factors, int maxIter, double w0, double alpha, double reg,
	double init_mean, double init_stdev, bool showProgress, bool showLoss, int userCount,
	int itemCount)
{
	this->trainMatrix = trainMatrix;
	//trainMatrix_R = _trainMatrix_R;
	this->testRatings = testRatings;
	this->topK = topK;
	this->factors = factors;
	this->maxIter = maxIter;
	this->w0 = w0;
	this->reg = reg;
	this->alpha = alpha;
	this->init_mean = init_mean;
	this->init_stdev = init_stdev;
	this->showloss = showLoss;
	this->showprogress = showProgress;
	this->itemCount = itemCount;
	this->userCount = userCount;
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
		p[i] = trainMatrix.outerIndexPtr()[i + 1] - trainMatrix.outerIndexPtr()[i];
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
	W.resize(userCount, itemCount);
	std::vector<T_d> tripletList;
	for (int u = 0; u < userCount; u++)
		for (int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u + 1]; i++)
			tripletList.push_back(T_d(u, trainMatrix_R.innerIndexPtr()[i], 1));
	W.setFromTriplets(tripletList.begin(), tripletList.end());

	//Init model parameters
	U = DenseMat(userCount, factors);
	V = DenseMat(itemCount, factors);

	U.init(init_mean, init_stdev);
	V.init(init_mean, init_stdev);
	initS();

}

void MF_fastALS::setTrain(SparseMat trainMatrix) {
	this->trainMatrix = new SparseMat(trainMatrix);
	W = new SparseMat(userCount, itemCount);
	for (int u = 0; u < userCount; u++)
		for (int i : this->trainMatrix.getRowRef(u).indexList())
			W.setValue(u, i, 1);
}

void MF_fastALS::setUV(DenseMat U, DenseMat V) {
	this->U = U.clone();
	this->V = V.clone();
	initS();
}

void MF_fastALS::buildModel() {
	double loss_pre = DBL_MAX;
	for (int iter = 0; iter < maxIter; iter++) {
		clock_t start = clock();
		for (int u = 0; u < userCount; u++) {
			update_user(u);

		}
		// Update item latent vectors
		for (int i = 0; i < itemCount; i++) {
			update_item(i);
		}

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
	std::cout << "Iter=" << iter << " " << (end - start)/CLOCKS_PER_SEC << " " << symbol << " loss:" << loss_cur << " " <<(clock() - end)/ CLOCKS_PER_SEC << std::endl;

	return loss_cur;
}

double MF_fastALS::loss() {
	double L = reg * (U.squaredNorm() + V.squaredNorm());
	for (int u = 0; u < userCount; u++) {
		double l = 0;
		std::vector<int> itemList;
		for (int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u + 1]; i++)
			itemList.push_back(trainMatrix_R.innerIndexPtr()[i]);
		for (int i : itemList) {
			double pred = predict(u, i);
			l += W.coeffRef(u, i) * pow(trainMatrix_R.coeffRef(u, i) - pred, 2);
			l -= Wi[i] * pow(pred, 2);
		}
		l += U.row(u) * SV.getData * U.row(u).transpose();
		L += l;
	}

	return L;
}

double MF_fastALS::predict(int u, int i) {
	return U.row(u) * V.row(i).transpose();
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
	std::vector<int> itemList;
	for (int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u + 1]; i++)
		itemList.push_back(trainMatrix_R.innerIndexPtr()[i]);
	if (itemList.size() == 0)        return;    // user has no ratings
	// prediction cache for the user

	for (int i : itemList) {
		prediction_items[i] = predict(u, i);
		rating_items[i] = trainMatrix_R.coeffRef(u, i);
		w_items[i] = W.coeffRef(u, i);
	}

	DenseVec oldVector = U.row(u);
	for (int f = 0; f < factors; f++) {
		double numer = 0, denom = 0;
		// O(K) complexity for the negative part
		for (int k = 0; k < factors; k++) {
			if (k != f)
				numer -= U.get(u, k) * SV.get(f, k);
		}
		//numer *= w0;
		// O(Nu) complexity for the positive part
		for (int i : itemList) {
			prediction_items[i] -= U.get(u, f) * V.get(i, f);
			numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * V(i, f);
			denom += (w_items[i] - Wi[i]) * V.get(i, f) * V.get(i, f);
		}
		denom += SV.get(f, f) + reg;

		// Parameter Update
		U.set(u, f, numer / denom);

		// Update the prediction cache
		for (int i : itemList)
			prediction_items[i] += U.get(u, f) * V.get(i, f);
	} // end for f

	// Update the SU cache
	for (int f = 0; f < factors; f++) {
		for (int k = 0; k <= f; k++) {
			double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k)
				+ U.get(u, f) * U.get(u, k);
			SU.set(f, k, val);
			SU.set(k, f, val);
		}
	}
}

void MF_fastALS::update_item(int i) {
	std::vector<int> userList;
	for (int j = trainMatrix.outerIndexPtr()[i]; j < trainMatrix.outerIndexPtr()[i + 1]; j++)
		userList.push_back(trainMatrix.innerIndexPtr()[j]);
	if (userList.size() == 0)        return; // item has no ratings.
	// prediction cache for the item
	for (int u : userList) {
		prediction_users[u] = predict(u, i);
		rating_users[u] = trainMatrix.coeffRef(u, i);
		w_users[u] = W.coeffRef(u, i);
	}


	DenseVec oldVector = V.row(i);
	for (int f = 0; f < factors; f++) {
		// O(K) complexity for the w0 part
		double numer = 0, denom = 0;
		for (int k = 0; k < factors; k++) {
			if (k != f)
				numer -= V.get(i, k) * SU.get(f, k);
		}
		numer *= Wi[i];

		// O(Ni) complexity for the positive ratings part
		for (int u : userList) {
			prediction_users[u] -= U.get(u, f) * V.get(i, f);
			numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U(u, f);
			denom += (w_users[u] - Wi[i]) * U.get(u, f) * U.get(u, f);
		}
		denom += Wi[i] * SU.get(f, f) + reg;

		// Parameter update
		V(i, f) = numer / denom;
		// Update the prediction cache for the item
		for (int u : userList)
			prediction_users[u] += U.get(u, f) * V.get(i, f);
	} // end for f

	// Update the SV cache
	for (int f = 0; f < factors; f++) {
		for (int k = 0; k <= f; k++) {
			double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * Wi[i]
				+ V.get(i, f) * V.get(i, k) * Wi[i];
			SV.set(f, k, val);
			SV.set(k, f, val);
		}
	}
}

void MF_fastALS::initS() {
	SU = DenseMat(U.transpose()*U.getData);
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
	result[0] = getHitRatio(rankList, gtItem);
	result[1] = getNDCG(rankList, gtItem);
	result[2] = getPrecision(rankList, gtItem);
	return result;
}

MF_fastALS::~MF_fastALS()
{
}
