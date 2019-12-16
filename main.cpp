#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <assert.h>
#include "MF_fastALS.cuh"
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <time.h>
#include "Rating.h"
#include <unordered_map>
#include "SparseMat.h"
#include "DenseMat.h"
#include "DenseVec.h"


int topK = 10;
int userCount;
int itemCount;
int user_id;
int item_id;

bool LessSort(Rating a, Rating b) { return(a.timestamp < b.timestamp); }
bool LessSort2(Rating a, Rating b) { return(a.userId < b.userId); }

void evaluate_model(MF_fastALS fals, std::vector<Rating> testRatings){
	std::vector<float> hits;
	std::vector<float> ndcgs;
	std::vector<float> precs;
	hits.resize(userCount);
	ndcgs.resize(userCount);
	precs .resize(userCount);
	//begin evaluation
	for (int u = 0; u < userCount; u++) {
		std::vector<float> result(3);
		int gtItem = testRatings[u].itemId;
		result = fals.evaluate_for_user(u, gtItem, topK);
		hits[u] = result[0];
		ndcgs[u] = result[1];
		precs[u] = result[2];

	}
	float res[3];
	res[0] = std::accumulate(std::begin(hits), std::end(hits), 0.0) / hits.size();
	res[1] = std::accumulate(std::begin(ndcgs), std::end(ndcgs), 0.0) / ndcgs.size();
	res[2] = std::accumulate(std::begin(precs), std::end(precs), 0.0) / precs.size();

	std::cout << "<hr, ndcg, prec>: \t" << res[0] << "\t" << res[1] << "\t" << res[2] << std::endl;
}



std::vector<std::vector<Rating>> ReadRatings_HoldOneOut(std::string dir) {

	std::cout << "Holdone out splitting" << std::endl;
	std::cout << "Sort items for each user." << std::endl;
	clock_t start = clock();
	std::vector<std::vector<Rating>> user_ratings;

	std::ifstream  fin;
	fin.open("amazon.rating");
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
	std::cout << "Sorting time:" << (float)(end - start) / CLOCKS_PER_SEC << std::endl;

	fin.close();
	return user_ratings;
}

struct IndexCountMap{
	int index;
	int count;
};

bool compareUser(IndexCountMap m1, IndexCountMap m2){
	return m1.count > m2.count;
}


int main(int argc, const char * argv[]) {
	std::string dataset_name = "yelp.rating";
	std::string method = "FastALS";
	float w0 = 10;
	bool showProgress = false;
	bool showLoss = true;
	int factors = 64;
	int maxIter = 20;
	float reg = 0.01;
	float alpha = 0.75;
	float init_mean = 0; 
	float init_stdev = 0.01;
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
  
    vector<map<int, float>> user_no_repeat;
    user_no_repeat.resize(userCount);
    vector<map<int, float>> item_no_repeat;
    item_no_repeat.resize(itemCount);
	float every_score;
	//reorder in both decreasing user number and item number
	
	IndexCountMap reorder_user[userCount];
	IndexCountMap reorder_item[itemCount];
	int map_user[userCount];
	int map_item[itemCount];
	
	for(int i=0; i<userCount; i++){
		reorder_user[i].index = i;
		reorder_user[i].count = 0;
	}	
	for(int i=0; i<itemCount; i++){
		reorder_item[i].index = i;	
		reorder_item[i].count = 0;	
	}	
	for (int u = 0; u < userCount; u++) {
		std::vector<Rating> rating = user_ratings[u];
		for (int i = (int)rating.size() - 1; i >= 0; i--) {
			user_id = rating[i].userId;
			item_id = rating[i].itemId;
			reorder_user[user_id].count++;
			reorder_item[item_id].count++;
		}
	}
	sort(reorder_user, reorder_user+userCount, compareUser);
	//sort(reorder_item, reorder_item+itemCount, compareitem);
	for(int i=0; i<userCount; i++){
		map_user[reorder_user[i].index] = i;
	}	
	for(int i=0; i<itemCount; i++){
		map_item[reorder_item[i].index] = i;
	}	
	

	//
	for (int u = 0; u < userCount; u++) {
		std::vector<Rating> rating = user_ratings[u];
		for (int i = (int)rating.size() - 1; i >= 0; i--) {
			//user_id = map_user[rating[i].userId];
			//item_id = map_item[rating[i].itemId];
			user_id = rating[i].userId;
			item_id = rating[i].itemId;
			every_score = rating[i].score;
			if (i == rating.size() - 1) { // test
				testRatings.push_back(Rating(user_id, item_id, every_score, 0));
			}
			else { 
        		user_no_repeat[user_id].insert(pair<int, float>(item_id, 1));
        		item_no_repeat[item_id].insert(pair<int, float>(user_id, 1));
			}
		}
	}
	
	//parition users into blocks and make it balance
	int block_num = 512;
	//IndexCountMap reorder_user[userCount];
	int update_index[block_num+1];
	
	int sum_index[block_num];
	vector<int> contain_index[block_num];
	for (int u = 0; u < userCount; u++){
		reorder_user[u].index = u;
		reorder_user[u].count = user_no_repeat[u].size();
	}
	
	//sort(reorder_user, reorder_user+userCount, compareUser);
	for(int i=0; i<block_num; i++){
		contain_index[i].push_back(i);
		sum_index[i] = reorder_user[i].count;
	}
	int user_index = block_num;
	int min_index;
	int min_value;
	while (user_index < userCount){
		min_index = 0;
		min_value = sum_index[0];
		for(int i=1; i<block_num; i++){
			if(sum_index[i] < min_value){
				min_index = i;
				min_value = sum_index[i];
			}
		}
		contain_index[min_index].push_back(user_index);
		sum_index[min_index] += reorder_user[user_index].count;
		user_index++;
	}
	
	int map_user2[userCount];
	user_index = 0;
	update_index[0]=0;
	int test_sum = 0;
	for(int i=0; i<block_num; i++){
		update_index[i+1] = update_index[i] + contain_index[i].size();
		//cout<<update_index[i]<<" ";
		for(int j=0; j<contain_index[i].size(); j++){
			map_user2[contain_index[i][j]] = user_index;
			user_index++;
			//test_sum+=reorder_user[contain_index[i][j]].count;
		}
		//std::cout<<test_sum<<" ";
	}

	int new_index = 0;
  	for (int u = 0; u < userCount; u++) {
		new_index = map_user[u];
		//new_index = u;
		trainMatrix.rows[new_index].setLength(user_no_repeat[new_index].size());
	}
	//std::cout<<sumsum<<"\n";
  	for (int i = 0; i < itemCount; i++) {
    	trainMatrix.cols[i].setLength(item_no_repeat[i].size());
    }
  	for (int u = 0; u < userCount; u++) {
    	map<int, float>::iterator iter;
		new_index = map_user[u];
		//new_index = u;
    	for(iter = user_no_repeat[new_index].begin(); iter != user_no_repeat[new_index].end(); iter++){
       		trainMatrix.setValue(new_index, iter->first, iter->second );
       	}
	}

	std::cout << "Generated splitted matrices time:" << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Data\t" << dataset_name << std::endl;
	std::cout << "#Users\t" << userCount << std::endl;
	std::cout << "#items\t" << itemCount << std::endl;
	std::cout << "#Ratings\t" << trainMatrix.itemCount() << "\t" << "tests\t" << testRatings.size() << std::endl;
	std::cout << "==========================================" << std::endl;

	assert(userCount == testRatings.size());
	sort(testRatings.begin(), testRatings.end(), LessSort2);
	for (int u = 0; u < userCount; u++)
		assert(u == testRatings[u].userId);

	MF_fastALS fals(trainMatrix, testRatings, topK, threadNum, factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss, userCount, itemCount, update_index);

	std::cout << "Start building model" << std::endl;
	fals.buildModel();
  	float res = fals.Calculate_RMSE();
  	std::cout<<"Evaluation loss: "<<res<<std::endl;

	return 0;
}
