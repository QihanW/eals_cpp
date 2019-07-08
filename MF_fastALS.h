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
	double reg = 0.01; 	// regularization parameters
	double w0 = 1;
	double init_mean = 0;  // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V
	int itemCount;
	int userCount;
	int topK;
	double alpha;
	double w_new = 1; // weight of new instance in online learning

	DenseMat U; // latent vectors for users
	DenseMat V;	// latent vectors for items
	DenseMat SU;
	DenseMat SV;
	SparseMat trainMatrix;
	SparseMat trainMatrix_R;
	SparseMat W;  // weight for each positive instance in trainMatrix

	std::vector<Rating> testRatings;
	//double * prediction_users;
	//double * prediction_items;
  //double * rating_users;
  //double *rating_items;
	//double * w_users; 
	//double * w_items;
	double * Wi; // weight for negative instances on item i.

	bool showprogress;
	bool showloss;
	

	MF_fastALS(SparseMat trainMatrix, std::vector<Rating> testRatings,
		int topK, int threadNum, int factors, int maxIter, double w0, double alpha, double reg,
		double init_mean, double init_stdev, bool showProgress, bool showLoss, int userCount,
		int itemCount);
	void setTrain(SparseMat trainMatrix);
	void setUV(DenseMat U, DenseMat V);
	void buildModel();
	void runOneIteration();
	double showLoss(int iter, double time, double loss_pre);
	double loss();
	double predict(int u, int i);
	void updateModel(int u, int i);
	double getHitRatio(std::vector<int> rankList, int gtItem);
	double getNDCG(std::vector<int> rankList, int gtItem);
	double getPrecision(std::vector<int> rankList, int gtItem);
	std::vector<double> evaluate_for_user(int u, int gtItem, int topK);
	void update_user_thread(int u);
	void update_user_SU(double *oldVector, double *uget);
	void update_item_thread(int i);
	void update_item_SV(int i, double *oldVector, double *vget);
	~MF_fastALS();

protected: 
 // void update_user(int u);
	void update_item(int i);
private:
	void initS();

};


#endif // !MF_FASTALS_H

