#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <assert.h>
#include "MF_fastALS.h"
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <time.h>
#include "Rating.h"
#include <unordered_map>
#include "SparseMat.h"
#include "DenseMat.h"
#include "DenseVec.h"
/*
using namespace Eigen;
typedef SparseMatrix<double> SpMat;
typedef SparseMatrix<double, RowMajor> SpMat_R;
typedef Triplet<int> T;
typedef Matrix<double, Dynamic, 1> VectorXd;
*/

int topK = 10;
int userCount;
int itemCount;
int user_id;
int item_id;



bool LessSort(Rating a, Rating b) { return(a.timestamp < b.timestamp); }

void evaluate_model(MF_fastALS fals, std::vector<Rating> testRatings){
  std::vector<double> hits;
  std::vector<double> ndcgs;
  std::vector<double> precs;

	hits.resize(userCount);
	ndcgs.resize(userCount);
	precs .resize(userCount);
	//begin evaluation
	for (int u = 0; u < userCount; u++) {
		std::vector<double> result(3);
		int gtItem = testRatings[u].itemId;
		result = fals.evaluate_for_user(u, gtItem, topK);
		hits[u] = result[0];
		ndcgs[u] = result[1];
		precs[u] = result[2];

	}
	double res[3];
	//    VectorXd hits;
	//    VectorXd ndcgs;
	//    VectorXd precs;
	//
	res[0] = std::accumulate(std::begin(hits), std::end(hits), 0.0) / hits.size();
	res[1] = std::accumulate(std::begin(ndcgs), std::end(ndcgs), 0.0) / ndcgs.size();
	res[2] = std::accumulate(std::begin(precs), std::end(precs), 0.0) / precs.size();

	std::cout << "<hr, ndcg, prec>: \t" << res[0] << "\t" << res[1] << "\t" << res[2] << std::endl;
}

/*
void ReadRatings_GlobalSplit(std::string dir) {
	
}
*/

std::vector<std::vector<Rating>> ReadRatings_HoldOneOut(std::string dir) {

	std::cout << "Holdone out splitting" << std::endl;
	std::cout << "Sort items for each user." << std::endl;
	clock_t start = clock();
	std::vector<std::vector<Rating>> user_ratings;
	//std::cout << dir << std::endl;

	std::ifstream  fin;
	fin.open("yelp.rating");
	std::string line;
	
	if (!fin.is_open()) {
		fprintf(stderr, "Error: cannot open the file %s\n", dir.c_str());
		exit(EXIT_FAILURE);
	}

	std::string line2;
	
	float score;
	long timestamp = 0;
	int x = 0;

	while (std::getline(fin, line)) {
		//std::getline(fin, line);
		std::istringstream word(line);
		word >> user_id;
		word >> item_id;
		word >> score;
		word >> timestamp;

		Rating rating(user_id,
			item_id,
			score,
			timestamp);
		if (user_ratings.size() < rating.userId + 1) {
			user_ratings.push_back(std::vector<Rating>());
     // std::cout << rating.userId << std::endl;
		}
		user_ratings.at(rating.userId).push_back(rating);
		userCount = fmax(userCount, rating.userId);
		itemCount = fmax(itemCount, rating.itemId);
		x++;
	}
  std::cout<<"line num of yelp: "<<x<<std::endl;
	userCount++;
	itemCount++;
	assert(userCount == user_ratings.size());

	for (int u = 0; u < userCount; u++) {
		sort(user_ratings[u].begin(), user_ratings[u].end(), LessSort);
	}
	clock_t end = clock();
	std::cout << "Sorting time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

	fin.close();
	return user_ratings;
}

int main(int argc, const char * argv[]) {
	std::string dataset_name = "yelp.rating";
	std::string method = "FastALS";
	double w0 = 10;
	bool showProgress = false;
	bool showLoss = true;
	int factors = 64;
	int maxIter = 50;
	double reg = 0.01;
	double alpha = 0.75;
	double init_mean = 0; 
	double init_stdev = 0.01;
	int threadNum = 1;
	/*
	if (argc > 0) {
		dataset_name = argv[0];
		method = argv[1];
		w0 = std::stod(argv[2]);
		if(argv[3]=="true") showProgress = true;
		if (argv[4] == "false") showLoss = false;
		factors = std::stoi(argv[5]);
		maxIter = std::stoi(argv[6]);
		reg = std::stod(argv[7]);
		alpha = std::stod(argv[8]);
	}
	*/

	std::vector<std::vector<Rating>> user_ratings;
	user_ratings = ReadRatings_HoldOneOut(dataset_name);

	std::cout << "Generate rating matrices" << std::endl;
	std::vector<Rating> testRatings;
	clock_t start = clock();
	SparseMat trainMatrix(userCount, itemCount);
  
  int num = 0;
  vector<map<int, double>> user_no_repeat;
  user_no_repeat.resize(userCount);
  vector<map<int, double>> item_no_repeat;
  item_no_repeat.resize(itemCount);
	//std::vector<T> tripletList;
	for (int u = 0; u < userCount; u++) {
		std::vector<Rating> rating = user_ratings[u];
		for (int i = (int)rating.size() - 1; i >= 0; i--) {
			user_id = rating[i].userId;
			item_id = rating[i].itemId;
			if (i == rating.size() - 1) { // test
				testRatings.push_back(rating[i]);
			}
			else { // train
				//num++;
				//trainMatrix.setValue(user_id, item_id, 1);
        user_no_repeat[user_id].insert(pair<int, double>(item_id, 1));
        item_no_repeat[item_id].insert(pair<int, double>(user_id, 1));
			}
			//                trainMatrix.insert(user_id, item_id) =  1;
		}
		num += rating.size()-1 - user_no_repeat[u].size();
	}

	for (int u = 0; u < userCount; u++) {
		trainMatrix.rows[u].setLength(user_no_repeat[u].size());
  }
  for (int i = 0; i < itemCount; i++) {
    trainMatrix.cols[i].setLength(item_no_repeat[i].size());
  }
  for (int u = 0; u < userCount; u++) {
    map<int, double>::iterator iter;
    //int len = user_no_repeat[u].size();
    for(iter = user_no_repeat[u].begin(); iter != user_no_repeat[u].end(); iter++){
       trainMatrix.setValue(u, iter->first, iter->second );
       
    }
  }

	/*
	num = 0;
  for (int u = 0; u < itemCount; u++){
    num += trainMatrix.cols[u].spv.size();
  }*/
  std::cout<<"Num of elements: "<<num<<std::endl;
  //std::cout << trainMatrix.rows.size()<<endl;
	//    trainMatrix.makeCompressed();
	std::cout << "Generated splitted matrices time:" << (double)(clock() - start) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Data\t" << dataset_name << std::endl;
	std::cout << "#Users\t" << userCount << std::endl;
	std::cout << "#items\t" << itemCount << std::endl;
	std::cout << "#Ratings\t" << trainMatrix.itemCount() << "\t" << "tests\t" << testRatings.size() << std::endl;

	std::cout << "==========================================" << std::endl;

	assert(userCount == testRatings.size());
	for (int u = 0; u < userCount; u++)
		assert(u == testRatings[u].userId);

	MF_fastALS fals(trainMatrix, testRatings, topK, threadNum, factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss, userCount, itemCount);

	std::cout << "Start building model" << std::endl;
	fals.buildModel();
	evaluate_model(fals, testRatings);

	return 0;
}
