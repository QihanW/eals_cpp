#ifndef MF_FASTALS_H
#define MF_FASTALS_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <vector>
#include "DenseVec.h"
#include "DenseMat.h"
#include "SparseMat.h"
#include "Rating.h"

//using namespace std;

class MF_fastALS
{
public:

	int factors = 10; 	// number of latent factors.
	int maxIter = 500; 	// maximum iterations.
	float reg = 0.01; 	// regularization parameters
	float w0 = 1;
	float init_mean = 0;  // Gaussian mean for init V
	float init_stdev = 0.01; // Gaussian std-dev for init V
	int itemCount;
	int userCount;
	int topK;
	float alpha;
	float w_new = 1; // weight of new instance in online learning

	DenseMat U; // latent vectors for users
	DenseMat V;	// latent vectors for items
	DenseMat SU;
	DenseMat SV;
	SparseMat trainMatrix;
	SparseMat trainMatrix_R;
	SparseMat W;  // weight for each positive instance in trainMatrix

	float *Wi;
	
	//replace U, V, SV
	float *u_values;
	float *v_values;
	float *sv_values;

	std::vector<Rating> testRatings;
  	float su_one[256];                                              
  	float sv_one[256]; 
	bool showprogress;
	bool showloss;
	//index
	int *partition_index;
	

	MF_fastALS(SparseMat trainMatrix, std::vector<Rating> testRatings,
		int topK, int threadNum, int factors, int maxIter, float w0, float alpha, float reg,
		float init_mean, float init_stdev, bool showProgress, bool showLoss, int userCount,
		int itemCount, int update_index[]);
	void setTrain(SparseMat trainMatrix);
	void setUV(DenseMat U, DenseMat V);
	void buildModel();
	void runOneIteration();
	float showLoss(int iter, float time, float loss_pre);
	float loss();
	float predict(int u, int i);
	void updateModel(int u, int i);
	float getHitRatio(std::vector<int> rankList, int gtItem);
	float getNDCG(std::vector<int> rankList, int gtItem);
	float getPrecision(std::vector<int> rankList, int gtItem);
	std::vector<float> evaluate_for_user(int u, int gtItem, int topK);
	void update_user_thread(int u);
	void update_user_SU(float *oldVector, float *uget);
	void update_item_thread(int i);
	void update_item_SV(int i, float *oldVector, float *vget);
	float Calculate_RMSE();
	void updateUserSchedule1();
	void updateUserSchedule2();
	void updateItemSchedule();
	void update_user_cpu(int u, float *u_numer_h, float *u_denom_h);
	//void update_user_cpu(int u, float *u_numer_h, float *u_denom_h, float *preidiction_items, int *train_n);
	~MF_fastALS();

protected: 
 // void update_user(int u);
	void update_item(int i);
private:
	void initS();

};


#endif // !MF_FASTALS_H

