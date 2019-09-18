#include <stdio.h> 
#include <iostream>
using namespace std;

#define BLOCK_NUM 1                                                           
#define THREAD_NUM 4 

/*./run
void updateUserSchedule2(float **trainMatrixdo, float **W, float **U, int **trainMatrixin, float **V, float **SV, float *Wi);


__global__ void updateUserCuda(float **prediction_items, float **rating_items, float **w_items, int userCount, int factors, float reg, float **w_cu1, float **u_cu, float **v_cu, float *wi_cu, float **sv_cu, float **train_spvdo_cu, int *train_n_cu, int **train_spvin_cu, float **v_col){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_item;                                                              
  float ifv, ufget, tmp_uget;                                                 
  int i;                                                                      
  float res;                                                                  
  float numer, denom;                                                                                                                     
  //printf("1");
  //printf("%f ", prediction_items[4][0]);
  for(int u = bid*THREAD_NUM+tid; u < userCount; u+=BLOCK_NUM*THREAD_NUM){                                                                                    
   // printf("1");
    size_item = train_n_cu[u];                                                
    if (size_item == 0)        continue ;                                   
    for (int j = 0; j < size_item; j++) {                                     
      //printf("%d", j);
      i = train_spvin_cu[u][j];                                                        
      res = 0;                                                                
      for(int k=0; k<factors; k++){                                           
        res += u_cu[u][k] * v_cu[i][k];                         
      }                                                                       
     // printf("%f", res);
      //printf("%f", prediction_items[1][j]);
      prediction_items[u][j] = res;
      //printf("%f ", res);
      rating_items[u][j] = train_spvdo_cu[u][j];                                 
      w_items[u][j] = w_cu1[u][j];
      //printf("1");
    }                              
    //printf("1");
    for (int f = 0; f < factors; f++) {                                       
      numer = 0, denom = 0;                                                   
      for(int j = 0; j<size_item; j++){                                       
        i = train_spvin_cu[u][j];                                                      
        v_col[u][j] = v_cu[i][f];                                         
      }                                                                     
      for(int k = 0; k<factors; k++){                                         
        numer -= u_cu[u][k] * sv_cu[f][k];                        
      }                                                                         
     // printf("%f ", numer);
      ufget = u_cu[u][f];                                                
      for (int j = 0; j<size_item; j++) {                                       
        i = train_spvin_cu[u][j];                                                        
        ifv = v_col[u][j];                                                       
        prediction_items[u][j] -= ufget * ifv;                                     
        numer += (w_items[u][j] * rating_items[u][j] - (w_items[u][j] - wi_cu[i]) * prediction_items[u][j]) * ifv;
        denom += (w_items[u][j] - wi_cu[i]) * ifv * ifv;                           
      }                                                                       
      denom +=sv_cu[f][f] + reg;                                         
      u_cu[u][f] = numer / denom;                                        
      tmp_uget = numer / denom; 
      //printf("%f ", tmp_uget);
      //printf("1");
      for (int j = 0; j<size_item; j++){                                        
        prediction_items[u][j] += tmp_uget * v_col[u][j];                             
      }                                                                        
    }                                                          
  }                                                                           
}                 

__global__ void updateUserCuda2(float **prediction_items, float **rating_items, float **w_items, int userCount, int factors, float reg, float **w_cu1, float **u_cu, float **v_cu, float *wi_cu, float **sv_cu, float **train_spvdo_cu, int *train_n_cu, int **train_spvin_cu, float **v_col){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_item;
  float ifv, ufget, tmp_uget;
  int i;
  float res;
  float numer, denom;

  int id =  bid*THREAD_NUM+tid;

  int divide = userCount/(BLOCK_NUM*THREAD_NUM);
  int last = userCount%(BLOCK_NUM*THREAD_NUM);
  int dis = divide;
  int u=0;
  if(id<last){

    dis++;
    u = (id-1)*dis;
    for(int i = 0; i < dis; i++){

      size_item = train_n_cu[u];
      if (size_item == 0)        continue ;
      for (int j = 0; j < size_item; j++) {
        i = train_spvin_cu[u][j];
        res = 0;
        for(int k=0; k<factors; k++){
          res += u_cu[u][k] * v_cu[i][k];
        }
        prediction_items[u][j] = res;
        rating_items[u][j] = train_spvdo_cu[u][j];
        w_items[u][j] = w_cu1[u][j];
      }

      for (int f = 0; f < factors; f++) {
        numer = 0, denom = 0;
        for(int j = 0; j<size_item; j++){
          i = train_spvin_cu[u][j];
          v_col[u][j] = v_cu[i][f];
        }
        for(int k = 0; k<factors; k++){
          numer -= u_cu[u][k] * sv_cu[f][k];
        }
        ufget = u_cu[u][f];
        for (int j = 0; j<size_item; j++) {
          i = train_spvin_cu[u][j];
          ifv = v_col[u][j];

          prediction_items[u][j] -= ufget * ifv;
          numer += (w_items[u][j] * rating_items[u][j] - (w_items[u][j] - wi_cu[i]) * prediction_items[u][j]) * ifv;
          denom += (w_items[u][j] - wi_cu[i]) * ifv * ifv;

        }
        denom +=sv_cu[f][f] + reg;
        u_cu[u][f] = numer / denom;
        tmp_uget = numer / denom;
        //printf("%f ", tmp_uget);
        for (int j = 0; j<size_item; j++){
          prediction_items[u][j] += tmp_uget * v_col[u][j];
        }
      }
      u++;
    }
  }
  else{
    u = last * (dis+1) + (id-last)*dis;
    for(int i = 0; i < dis; i++){
      size_item = train_n_cu[u];
      if (size_item == 0)        continue ;
      for (int j = 0; j < size_item; j++) {
        i = train_spvin_cu[u][j];
        res = 0;
        for(int k=0; k<factors; k++){
          res += u_cu[u][k] * v_cu[i][k];
        }
        prediction_items[u][j] = res;
        rating_items[u][j] = train_spvdo_cu[u][j];
        w_items[u][j] = w_cu1[u][j];
      }

      for (int f = 0; f < factors; f++) {
        numer = 0, denom = 0;
        for(int j = 0; j<size_item; j++){
          i = train_spvin_cu[u][j];
          v_col[u][j] = v_cu[i][f];
        }
        for(int k = 0; k<factors; k++){
          numer -= u_cu[u][k] * sv_cu[f][k];
        }
        ufget = u_cu[u][f];
        for (int j = 0; j<size_item; j++) {
          i = train_spvin_cu[u][j];
          ifv = v_col[u][j];

          prediction_items[u][j] -= ufget * ifv;
          numer += (w_items[u][j] * rating_items[u][j] - (w_items[u][j] - wi_cu[i]) * prediction_items[u][j]) * ifv;
          denom += (w_items[u][j] - wi_cu[i]) * ifv * ifv;

        }
        denom +=sv_cu[f][f] + reg;
        u_cu[u][f] = numer / denom;
        tmp_uget = numer / denom;
        //printf("%f ", tmp_uget);
        for (int j = 0; j<size_item; j++){
          prediction_items[u][j] += tmp_uget * v_col[u][j];
        }
      }
    }
  }

__global__ void updateUserCuda(float *prediction_items, float *rating_items, float *w_items, float *v_col, int uborder, int vborder, int userCount, int factors, float reg, float *w_cu, float *uvsv_cu, float *wi_cu, float *train_spvdo_cu, int *train_n_cu, int *train_spvin_cu){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_item;
  float ifv, ufget, tmp_uget;
  int i;
  float res;
  float numer, denom;
  int size2;
  int index;
  for(int u = bid*THREAD_NUM+tid; u < userCount; u+=BLOCK_NUM*THREAD_NUM){
    size_item = train_n_cu[u+1] - train_n_cu[u];
    size2 = train_n_cu[u];
    if (size_item == 0)        continue ;
    for (int j = 0; j < size_item; j++) {
      index = size2+j;
      i = train_spvin_cu[index];
      res = 0;
      for(int k=0; k<factors; k++){
        res += uvsv_cu[u*factors+k] * uvsv_cu[uborder+i*factors+k];
      }
      prediction_items[index] = res;
      rating_items[index] = train_spvdo_cu[index];
      w_items[index] = w_cu[index];
    }
    for (int f = 0; f < factors; f++) {
      numer = 0, denom = 0;
      for(int j = 0; j<size_item; j++){
        i = train_spvin_cu[size2+j];
        v_col[size2+j] = uvsv_cu[uborder+i*factors+f];
      }
      for(int k = 0; k<factors; k++){
        numer -= uvsv_cu[u*factors+k] * uvsv_cu[vborder+f*factors+k];
      }
      ufget = uvsv_cu[u*factors+f];
      for (int j = 0; j<size_item; j++) {
        index = size2+j;
        i = train_spvin_cu[index];
        ifv = v_col[index];
        prediction_items[index] -= ufget * ifv;
        numer += (w_items[index] * rating_items[index] - (w_items[index] - wi_cu[i]) * prediction_items[index]) * ifv;
        denom += (w_items[index] - wi_cu[i]) * ifv * ifv;
      }
      denom += uvsv_cu[vborder+f*factors+f] + reg;
      uvsv_cu[u*factors+f] = numer / denom;
      tmp_uget = numer / denom;
      printf("%f ", tmp_uget);
      for (int j = 0; j<size_item; j++){
        prediction_items[size2+j] += tmp_uget * v_col[size2+j];
      }
    }
  }
//=======
//>>>>>>> da1f5daa4a28ef05be26223dd8ce61c51ef97c94
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

void updateUserSchedule1(){
int userCount = 10;
int itemCount = 20;
int factors = 8;
float reg = 0;

int max_size = 5, size;
float *prediction_items;
float *rating_items;
float *w_items, *v_col;
prediction_items = (float *)malloc(sizeof(float)*max_size);
rating_items = (float *)malloc(sizeof(float)*max_size);
w_items = (float *)malloc(sizeof(float)*max_size);
v_col = (float *)malloc(sizeof(float)*max_size);                                                                    

float **w_cu, **u_cu, **v_cu, **sv_cu, **train_spvdo_cu, **u2;
int **train_spvin_cu;
w_cu = (float **)malloc(sizeof(float *)*userCount);
u_cu = (float **)malloc(sizeof(float *)*userCount);
u2 = (float **)malloc(sizeof(float *)*userCount);
v_cu = (float **)malloc(sizeof(float *)*itemCount);
sv_cu = (float **)malloc(sizeof(float *)*factors);
train_spvin_cu = (int **)malloc(sizeof(int *)*userCount);
train_spvdo_cu = (float **)malloc(sizeof(float *)*userCount);

int *train_n_cu;
train_n_cu = (int *)malloc(sizeof(int)*userCount);

for (int u = 0; u < userCount; u++){
  size = 5;
  w_cu[u] = (float *)malloc(sizeof(float)*size);
  train_spvin_cu[u] = (int *)malloc(sizeof(int)*size);
  train_spvdo_cu[u] = (float *)malloc(sizeof(float)*size);
  for(int i = 0; i < size; i++){
    w_cu[u][i] = 0.5;
    train_spvin_cu[u][i] = i;
    train_spvdo_cu[u][i] = 1;
  }
  train_n_cu[u] = size;
  u_cu[u] = (float *)malloc(sizeof(float)*factors);
  u2[u] = (float *)malloc(sizeof(float)*factors);
  for(int i = 0; i < factors; i++){
    u_cu[u][i] = i;
    u2[u][i] = u_cu[u][i];
  }
  }


  for (int u = 0; u < itemCount; u++){
    v_cu[u] = (float *)malloc(sizeof(float)*factors);
    for(int i = 0; i < factors; i++)
      v_cu[u][i] = 1;
  }
  for (int u = 0; u < factors; u++){
    sv_cu[u] = (float *)malloc(sizeof(float)*factors);
    for(int i = 0; i < factors; i++)
      sv_cu[u][i] = 2;
  }

  float *wi_cu;
  wi_cu = (float *)malloc(sizeof(float)*itemCount);
  for(int i=0; i<itemCount; i++){
    wi_cu[i] = ((float)i)/5;
  }

  updateUserCpp(prediction_items, rating_items, w_items, userCount, factors, reg, w_cu, u_cu, v_cu, wi_cu, sv_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, v_col);
//<<<<<<< HEAD
  //updateUserSchedule2(train_spvdo_cu, w_cu, u2, train_spvin_cu, v_cu, sv_cu, wi_cu);
//=======
  //updateUserSchedule2(train_spvdo_cu, w_cu, u2, train_spvin_cu, v_cu, sv_cu, wi_cu);
//>>>>>>> da1f5daa4a28ef05be26223dd8ce61c51ef97c94

  cout<<"cpp: "<<endl;
  //float sum = 0;
  for(int u=0; u<userCount; u++){
    for(int i=0; i<factors; i++)
    {   cout<<u_cu[u][i]<<" ";
        //sum += u_cu[u][i];
    }
  }
  cout<<endl;
//<<<<<<< HEAD
  cout<<"cuda :";
  updateUserSchedule2(train_spvdo_cu, w_cu, u2, train_spvin_cu, v_cu, sv_cu, wi_cu);
  cout<<endl;
  //cout<<"sum: "<<sum<<endl;
//=======
//cout<<"sum: "<<sum<<endl;
//>>>>>>> da1f5daa4a28ef05be26223dd8ce61c51ef97c94
  free(wi_cu);
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
    free(u2[u]);
  }
  for (int u = 0; u < itemCount; u++){
    free(v_cu[u]);
  }
  for (int u = 0; u < factors; u++){
    free(sv_cu[u]);
  }
  
  free(w_cu);
  free(u_cu);
  free(u2);
  free(v_cu);
  free(sv_cu);
  free(train_spvin_cu);                                                                                               
  free(train_spvdo_cu);
}
//<<<<<<< HEAD
void updateUserSchedule2(float **trainMatrixdo, float **W, float **U, int **trainMatrixin, float **V, float **SV, float *Wi){
  int userCount = 10;                                                           
  int itemCount = 20;                                                           
  int factors = 8;                                                              
  float reg = 0;                                                                
  int max_size = 5, size, size2;
  float *w_items, *prediction_items, *v_col, *rating_items;
  int total_size = max_size * userCount + 10;
  cudaMalloc((void**)&prediction_items,sizeof(float)*total_size);
  cudaMalloc((void**)&w_items,sizeof(float)*total_size);
  cudaMalloc((void**)&rating_items,sizeof(float)*total_size);
  cudaMalloc((void**)&v_col,sizeof(float)*total_size);

  float *w_cu, *uvsv_cu, *train_spvdo_cu;
  int *train_spvin_cu;
  int uvsvSize = sizeof(float) * (userCount + itemCount + factors) * factors;
  cudaMalloc((void**)&w_cu, sizeof(float)*total_size);
  cudaMalloc((void**)&uvsv_cu, uvsvSize);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int)*total_size);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float)*total_size);
  int *train_n, *train_n_cu;
  train_n = (int *)malloc(sizeof(int)*userCount);
  cudaMalloc((void**)&train_n_cu,sizeof(int)*userCount);
  
  float *w_h, *uvsv_h, *train_spvdo_h;
  int *train_spvin_h;
  w_h = (float *)malloc(sizeof(float)*total_size);
  uvsv_h = (float *)malloc(uvsvSize);
  train_spvin_h = (int *)malloc(sizeof(int)*total_size);
  train_spvdo_h = (float *)malloc(sizeof(float)*total_size);
  train_n[0] = 0;
  //cout<<total_size<<endl;
  for (int u = 0; u < userCount; u++){
    size = 5;
    size2 = train_n[u];
    train_n[u+1] = size2 + size;
    for(int i=0; i<size; i++){
      //cout<<size2+i<<endl;
      w_h[size2+i] = W[u][i];
      train_spvin_h[size2+i] = trainMatrixin[u][i];
      train_spvdo_h[size2+i] = trainMatrixdo[u][i];
    }
  }
  int uborder = userCount * factors;
  int vborder = uborder + itemCount * factors;
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[u*factors+i] = U[u][i];
  } 
  for (int u = 0; u < itemCount; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[uborder+u*factors+i] = V[u][i];
  } 
  for (int u = 0; u < factors; u++){
    for(int i=0; i<factors; i++)
      uvsv_h[vborder+u*factors+i] = SV[u][i];
  } 
  cudaMemcpy(uvsv_cu, uvsv_h, uvsvSize, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int)*total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*userCount, cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);
  updateUserCuda<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, rating_items, w_items, v_col, uborder, vborder, userCount, factors, reg, w_cu, uvsv_cu, wi_cu, train_spvdo_cu, train_n_cu, train_spvin_cu);
  cudaMemcpy(uvsv_h, uvsv_cu, uvsvSize, cudaMemcpyDeviceToHost);
  for (int u = 0; u < userCount; u++){
    for(int i=0; i<factors; i++)
      U[u][i] = uvsv_h[u*factors+i];
  }
  cout<<endl;
  cudaFree(prediction_items);
  cudaFree(w_items);
  cudaFree(rating_items);
  cudaFree(v_col);
  //cout<<"470";
  cudaFree(train_n_cu);
  cudaFree(w_cu);
  cudaFree(uvsv_cu);
  //cout<<"474";
  cudaFree(train_spvin_cu);
  //cout<<"476";
  //cudaFree(train_spvdo_cu);
  //cout<<"478";
  cudaFree(wi_cu);
  //cout<<"480";
  free(train_n);
  free(w_h);
  free(uvsv_h);
  free(train_spvin_h);
  free(train_spvdo_h);
  cudaFree(train_spvdo_cu);
  //cout<<"481";
}

/*
void updateUserSchedule2(float **trainMatrixdo, float **W, float **U, int **trainMatrixin, float **V, float **SV, float *Wi){                                          
  int max_size = 5, size;                                                   
  float **prediction_items;                                                      
  float **rating_items;                                                          
  float **w_items, **v_col;                                                       
                                                            
  cudaMalloc((void**)&prediction_items,sizeof(float *)*userCount);                 
  cudaMalloc((void**)&rating_items,sizeof(float *)*userCount);                     
  cudaMalloc((void**)&w_items,sizeof(float *)*userCount);                          
  cudaMalloc((void**)&v_col,sizeof(float *)*userCount);

  float **t1, **t2, **t3, **t4;
  t1 = (float **)malloc(sizeof(float *)*userCount);
  t2 = (float **)malloc(sizeof(float *)*userCount);  
  t3 = (float **)malloc(sizeof(float *)*userCount);  
  t4 = (float **)malloc(sizeof(float *)*userCount);  
  float *mm;
  mm = (float *)malloc(sizeof(float)*max_size);
  for(int i=0; i<max_size; i++)
    mm[i] = 0;
  for(int u=0; u<userCount; u++){
    float *tmp1, *tmp2, *tmp3, *tmp4;
    
    cudaMalloc((void**)&tmp1,sizeof(float)*max_size);
    cudaMalloc((void**)&tmp2,sizeof(float)*max_size);
    cudaMalloc((void**)&tmp3,sizeof(float)*max_size);
    cudaMalloc((void**)&tmp4,sizeof(float)*max_size);
    cudaMemcpy(tmp1, mm, sizeof(float)*max_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp2, mm, sizeof(float)*max_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp3, mm, sizeof(float)*max_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp4, mm, sizeof(float)*max_size, cudaMemcpyHostToDevice);
    t1[u] = tmp1;
    t2[u] = tmp2;
    t3[u] = tmp3;
    t4[u] = tmp4;
  }
  cudaMemcpy(prediction_items, t1, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(rating_items, t2, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(w_items, t3, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(v_col, t4, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
                                      
  free(mm);
  float **w_cu, **u_cu, **v_cu, **sv_cu, **train_spvdo_cu;                      
  int **train_spvin_cu;                                                         
  cudaMalloc((void**)&w_cu,sizeof(float *)*userCount);                          
  cudaMalloc((void**)&u_cu,sizeof(float *)*userCount);                          
  cudaMalloc((void**)&v_cu,sizeof(float *)*itemCount);                          
  cudaMalloc((void**)&sv_cu,sizeof(float *)*factors);                           
  cudaMalloc((void**)&train_spvin_cu,sizeof(int *)*userCount);                  
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float *)*userCount);                                                               
  int *train_n, *train_n_cu;                                                    
  train_n = (int *)malloc(sizeof(int)*userCount);                               
  cudaMalloc((void**)&train_n_cu,sizeof(int)*userCount);                        
  
  float **w_h, **u_h, **v_h, **sv_h, **train_spvdo_h;                           
  int **train_spvin_h;                                                          
  w_h = (float **)malloc(sizeof(float *)*userCount);                            
  u_h = (float **)malloc(sizeof(float *)*userCount);                            
  v_h = (float **)malloc(sizeof(float *)*itemCount);                            
  sv_h = (float **)malloc(sizeof(float *)*factors);                             
  train_spvin_h = (int **)malloc(sizeof(int *)*userCount);                      
  train_spvdo_h = (float **)malloc(sizeof(float *)*userCount);                                                                  
  for (int u = 0; u < userCount; u++){                                          
    size = 5;                                               
    train_n[u] = size;                                                          
    float *tmp_train, *tmp_w, *tmp_u;                                           
    int *tmp;                                                                   
    cudaMalloc((void**)&tmp_train,sizeof(float)*size);                          
    cudaMalloc((void**)&tmp_w,sizeof(float)*size);                              
    cudaMalloc((void**)&tmp_u,sizeof(float)*factors);                           
    cudaMalloc((void**)&tmp,sizeof(int)*size);                 
    cudaMemcpy(tmp_train, trainMatrixdo[u], sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_w, W[u], sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_u, U[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, trainMatrixin[u], sizeof(int)*size, cudaMemcpyHostToDevice);
    
    w_h[u] = tmp_w;                                                             
    u_h[u] = tmp_u;                                                             
    train_spvin_h[u] = tmp;                                                     
    train_spvdo_h[u] = tmp_train;                                               
  }                                                                             
                                                
  for (int u = 0; u < itemCount; u++){                                          
    float *tmp_v;                                                               
    cudaMalloc((void**)&tmp_v,sizeof(float)*factors);                           
    cudaMemcpy(tmp_v, V[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    v_h[u] = tmp_v;                                                             
  }                                                                             
  for (int u = 0; u < factors; u++){                                            
    float *tmp_sv;                                                              
    cudaMalloc((void**)&tmp_sv,sizeof(float)*factors);                          
    cudaMemcpy(tmp_sv, SV[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    sv_h[u] = tmp_sv;                                                           
  }                                                                             
                                                   
  cudaMemcpy(u_cu, u_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);     
  cudaMemcpy(v_cu, v_h, sizeof(float *)*itemCount, cudaMemcpyHostToDevice);     
  cudaMemcpy(sv_cu, sv_h, sizeof(float *)*factors, cudaMemcpyHostToDevice);     
  cudaMemcpy(w_cu, w_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);     
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*userCount, cudaMemcpyHostToDevice);
  float *wi_cu;                                                                 
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);                         
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);   
  
  cout<<"cuda: "<<endl;
  updateUserCuda2<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, rating_items, w_items, userCount, factors, reg, w_cu, u_cu, v_cu, wi_cu, sv_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, v_col);
  
  //cout<<"cuda: "<<endl;
  cudaMemcpy(u_h, u_cu, sizeof(float *)*userCount, cudaMemcpyDeviceToHost); //?    
  //int dif = 0;                                                                  
  for (int u = 0; u < userCount; u++){                                                           
    cudaMemcpy(U[u], u_h[u], sizeof(float)*factors, cudaMemcpyDeviceToHost);
    for(int i=0; i<factors; i++)
      cout<<U[u][i]<<" ";
  }                         
  cout<<endl;
   
  for(int u=0; u<userCount; u++){
    cudaFree(t1[u]);
    cudaFree(t2[u]);
    cudaFree(t3[u]);
    cudaFree(t4[u]);
  }  
  free(t1); 
  free(t2);
  free(t3);
  free(t4);                        
  cudaFree(prediction_items);                                                   
  cudaFree(rating_items);                                                       
  cudaFree(w_items);                                                            
  cudaFree(v_col);                                                              
  cudaFree(train_n_cu);                                                         
  free(train_n);                                                                
  cudaFree(wi_cu);                                                              
  for (int u = 0; u < userCount; u++){                                          
    cudaFree(w_h[u]);                                                           
    cudaFree(train_spvin_h[u]);                                                 
    cudaFree(train_spvdo_h[u]);                                                 
    cudaFree(u_h[u]);                                                           
  }                                                                             
  for (int u = 0; u < itemCount; u++){                                          
    cudaFree(v_h[u]);                                                           
  }                                                                             
  for (int u = 0; u < factors; u++){                                            
    cudaFree(sv_h[u]);                                                          
  }                                                                             
  cudaFree(w_cu);                                                               
  cudaFree(u_cu);                                                               
  cudaFree(v_cu);                                                               
  cudaFree(sv_cu);                                                              
  cudaFree(train_spvin_cu);                                                     
  cudaFree(train_spvdo_cu);                                                     
  free(w_h);                                                                    
  free(u_h);                                                                    
  free(v_h);                                                                    
  free(sv_h);                                                                   
  free(train_spvin_h);                                                          
  free(train_spvdo_h);              

}

void MF_fastALS::updateUserSchedule3(float **trainMatrixdo, float **W, float **U, int **trainMatrixin, float **V, float **SV, float *Wi){
  int userCount = 10;
  int itemCount = 20;
  int factors = 8;

  int max_size = 5, size;
  float **prediction_items;
  float **rating_items;
  float **w_items, **v_col;

  cudaMalloc((void**)&prediction_items,sizeof(float *)*userCount);
  cudaMalloc((void**)&rating_items,sizeof(float *)*userCount);
  cudaMalloc((void**)&w_items,sizeof(float *)*userCount);
  cudaMalloc((void**)&v_col,sizeof(float *)*userCount);

  float **t1, **t2, **t3, **t3;
  t1 = (float *)malloc(sizeof(float)*userCount);
  t2 = (float *)malloc(sizeof(float)*userCount);
  t3 = (float *)malloc(sizeof(float)*userCount);
  t4 = (float *)malloc(sizeof(float)*userCount);

  for(int u=0; u<userCount; u++){
    cudaMalloc((void**)&t1[u],sizeof(float)*max_size);
    cudaMalloc((void**)&t2[u],sizeof(float)*max_size);
    cudaMalloc((void**)&t3[u],sizeof(float)*max_size);
    cudaMalloc((void**)&t4[u],sizeof(float)*max_size);
  }
  cudaMemcpy(prediction_items, t1[u], sizeof(float)*max_size, cudaMemcpyHostToDevice);
  cudaMemcpy(rating_items, t2[u], sizeof(float)*max_size, cudaMemcpyHostToDevice);
  cudaMemcpy(w_items, t3[u], sizeof(float)*max_size, cudaMemcpyHostToDevice);
  cudaMemcpy(v_col, t4[u], sizeof(float)*max_size, cudaMemcpyHostToDevice);

  float **w_cu, **u_cu, **v_cu, **sv_cu, **train_spvdo_cu;
  int **train_spvin_cu;
  cudaMalloc((void**)&w_cu,sizeof(float *)*userCount);
  cudaMalloc((void**)&u_cu,sizeof(float *)*userCount);
  cudaMalloc((void**)&v_cu,sizeof(float *)*itemCount);
  cudaMalloc((void**)&sv_cu,sizeof(float *)*factors);
  cudaMalloc((void**)&train_spvin_cu,sizeof(int *)*userCount);
  cudaMalloc((void**)&train_spvdo_cu,sizeof(float *)*userCount);
  int *train_n, *train_n_cu;
  train_n = (int *)malloc(sizeof(int)*userCount);
  cudaMalloc((void**)&train_n_cu,sizeof(int)*userCount);

  float **w_h, **u_h, **v_h, **sv_h, **train_spvdo_h;
  int **train_spvin_h;
  w_h = (float **)malloc(sizeof(float *)*userCount);
  u_h = (float **)malloc(sizeof(float *)*userCount);
  v_h = (float **)malloc(sizeof(float *)*itemCount);
  sv_h = (float **)malloc(sizeof(float *)*factors);
  train_spvin_h = (int **)malloc(sizeof(int *)*userCount);
  train_spvdo_h = (float **)malloc(sizeof(float *)*userCount);
  for (int u = 0; u < userCount; u++){
    size = 5;
    train_n[u] = size;
    float *tmp_train, *tmp_w, *tmp_u;
    int *tmp;
    cudaMalloc((void**)&tmp_train,sizeof(float)*size);
    cudaMalloc((void**)&tmp_w,sizeof(float)*size);
    cudaMalloc((void**)&tmp_u,sizeof(float)*factors);
    cudaMalloc((void**)&tmp,sizeof(int)*size);
    cudaMemcpy(tmp_train, trainMatrixdo[u], sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_w, W[u], sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_u, U[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, trainMatrixin[u], sizeof(int)*size, cudaMemcpyHostToDevice);

    w_h[u] = tmp_w;
    u_h[u] = tmp_u;
    train_spvin_h[u] = tmp;
    train_spvdo_h[u] = tmp_train;
  }

  for (int u = 0; u < itemCount; u++){
    float *tmp_v;
    cudaMalloc((void**)&tmp_v,sizeof(float)*factors);
    cudaMemcpy(tmp_v, V[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    v_h[u] = tmp_v;
  }
  for (int u = 0; u < factors; u++){
    float *tmp_sv;
    cudaMalloc((void**)&tmp_sv,sizeof(float)*factors);
    cudaMemcpy(tmp_sv, SV[u], sizeof(float)*factors, cudaMemcpyHostToDevice);
    sv_h[u] = tmp_sv;
  }

  cudaMemcpy(u_cu, u_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(v_cu, v_h, sizeof(float *)*itemCount, cudaMemcpyHostToDevice);
  cudaMemcpy(sv_cu, sv_h, sizeof(float *)*factors, cudaMemcpyHostToDevice);
  cudaMemcpy(w_cu, w_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvdo_cu, train_spvdo_h, sizeof(float *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(train_spvin_cu, train_spvin_h, sizeof(int *)*userCount, cudaMemcpyHostToDevice);
  cudaMemcpy(train_n_cu, train_n, sizeof(int)*userCount, cudaMemcpyHostToDevice);
  float *wi_cu;
  cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
  cudaMemcpy(wi_cu, Wi, sizeof(float)*itemCount, cudaMemcpyHostToDevice);

  updateUserCuda<<<BLOCK_NUM,THREAD_NUM>>>(prediction_items, rating_items, w_items, userCount, factors, reg, w_cu, u_cu, v_cu, wi_cu, sv_cu, train_spvdo_cu, train_n_cu, train_spvin_cu, v_col);

  cudaMemcpy(u_h, u_cu, sizeof(float *)*userCount, cudaMemcpyDeviceToHost); //?
  int dif = 0;
  for (int u = 0; u < userCount; u++){
    cudaMemcpy(U[u], u_h[u], sizeof(float)*factors, cudaMemcpyDeviceToHost);
  }

  for(int u=0; u<userCount; u++){
    cudaFree(t1[u]);
    cudaFree(t2[u]);
    cudaFree(t3[u]);
    cudaFree(t4[u]);
  }
  free(t1);
  free(t2);
  free(t3);
  free(t4);
  cudaFree(prediction_items);
  cudaFree(rating_items);
  cudaFree(w_items);
  cudaFree(v_col);
  cudaFree(train_n_cu);
  free(train_n);
  cudaFree(wi_cu);
  for (int u = 0; u < userCount; u++){
    cudaFree(w_h[u]);
    cudaFree(train_spvin_h[u]);
    cudaFree(train_spvdo_h[u]);
    cudaFree(u_h[u]);
  }
  for (int u = 0; u < itemCount; u++){
    cudaFree(v_h[u]);
  }
  for (int u = 0; u < factors; u++){
    cudaFree(sv_h[u]);
  }
  cudaFree(w_cu);
  cudaFree(u_cu);
  cudaFree(v_cu);
  cudaFree(sv_cu);
  cudaFree(train_spvin_cu);
  cudaFree(train_spvdo_cu);
  free(w_h);
  free(u_h);
  free(v_h);
  free(sv_h);
  free(train_spvin_h);
  free(train_spvdo_h);

}

*/
/*
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

void testUpdateSu(){
  int userCount = 10;
  int factors = 5;

  float U[userCount][factors];
  float u_clone[userCount][factors];
  float res_cpp[factors][factors];
  float res_cuda[factors][factors];

  for(int i=0; i<userCount; i++){
    for(int j=0; j<factors; j++){
      U[i][j] = i;
      u_clone[i][j] = i+j;
    }
  }

  //cpp version
  float tmp;
    for (int f = 0; f < factors; f++) {                                                                                           
	    for (int k = 0; k <= f; k++) {
        float val = 0;
	      #pragma omp parallel for reduction(+:val) 
	      for (int u = 0; u < userCount; u++){
		      tmp = 0 - u_clone[u][f] * u_clone[u][k] + U[u][f] * U[u][k];
          val += tmp;
        }
		    res_cpp[f][k] = val;                                                      
		    res_cpp[k][f] = val;                                                                                                                                 
      }     
    }

  //cuda version
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
      U_h[ii] = U[u][j];
      U_clone_h[ii] = u_clone[u][j];
      ii++;
    }
  }

  cudaMemcpy(U_clone_cu, U_clone_h, byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(U_cu, U_h, byteSize, cudaMemcpyHostToDevice);
    
    //int indexf, indexk;
    for (int f = 0; f < factors; f++) { 
      for (int k = 0; k <= f; k++) {                                          
        float val = 0;
        computeSUValue<<<BLOCK_NUM,THREAD_NUM,0>>>(U_clone_cu, U_cu, f, k, factors, result_cu, userCount);
        cudaMemcpy(result, result_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+:val)                          
        for (int u = 0; u < BLOCK_NUM; u++){                                  
          val += result[u];                                                       
        }                                                                     
        res_cuda[f][k] = val;
        res_cuda[k][f] = val;
       }                                                                  
    }

    cudaFree(U_clone_cu);
    cudaFree(U_cu);
    cudaFree(result_cu);
    cudaFreeHost(U_clone_h);
    cudaFreeHost(U_h);
    cudaFreeHost(result);

    for(int i=0; i<factors; i++){
      for(int j=0; j<factors; j++){
        std::cout<<"i: "<<i<<" j: "<<j<<" cpp: "<<res_cpp[i][j]<<" cuda: "<<res_cuda[i][j]<<endl;
      }
    }
}
*/
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
    //printf("%f %f %f %f\n", u_clone[index+f], u_clone[index+k], u[index+f], u[index+k]);
    //printf("%f ", temp);
    val += temp*wi_cu[i];
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

void testUpdateSv(){
  int itemCount = 10;
  int factors = 5;
  float V[itemCount][factors];
  float v_clone[itemCount][factors];
  float wii[itemCount];
  float res_cpp[factors][factors];
  float res_cuda[factors][factors];

  for(int i=0; i<itemCount; i++){
    for(int j=0; j<factors; j++){
      V[i][j] = 2;
      v_clone[i][j] = 1;
    }
    wii[i] = 0.5;
  }

  //cpp version
  float tmp;
    for (int f = 0; f < factors; f++) {
	    for (int k = 0; k <= f; k++) {
		    float val = 0;
 		    #pragma omp parallel for reduction(+:val)
  		  for (int u = 0; u < itemCount; u++){
 			    tmp = 0 - v_clone[u][f] * v_clone[u][k] + V[u][f] * V[u][k] ;
           //cout<<tmp<<" ";
           tmp = tmp *  wii[u];
          val += tmp;
          //cout<<v_clone[u][f]<<" "<<v_clone[u][k]<<" "<<V[u][f]<<" "<<V[u][k]<<"\n";
          //cout<<tmp<<" ";
		    }
        res_cpp[f][k] = val;
        res_cpp[k][f] = val;
      }
    }
   //cout<<"cpp is okay"<<endl; 

  int byteSize = sizeof(float)*itemCount*factors;
    float *V_clone_cu, *V_cu, *V_h, *V_clone_h, *resultv, *resultv_cu, *wi_cu;
    V_h = (float *)malloc(byteSize);
    V_clone_h = (float *)malloc(byteSize);
    resultv = (float *)malloc(sizeof(float)*(BLOCK_NUM));
    cudaMalloc((void**)&V_clone_cu, byteSize);
    cudaMalloc((void**)&V_cu, byteSize);
    cudaMalloc((void**)&wi_cu, sizeof(float)*itemCount);
    cudaMalloc((void**)&resultv_cu, sizeof(float)*(BLOCK_NUM));
    int ii = 0;

    #pragma omp parallel for
    for (int u = 0; u < itemCount; u++){
      for(int j = 0; j < factors; j++){
        V_h[ii] = V[u][j];
        V_clone_h[ii] = v_clone[u][j];
        ii++;
      }
    }
   //cout<<"1005 is okay"<<endl;
    cudaMemcpy(V_clone_cu, V_clone_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(V_cu, V_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(wi_cu, wii, sizeof(float)*itemCount, cudaMemcpyHostToDevice);

    for (int f = 0; f < factors; f++) {
      for (int k = 0; k <= f; k++) {
        float val = 0;
        computeSVValue<<<BLOCK_NUM,THREAD_NUM,0>>>(V_clone_cu, V_cu, f, k, factors, resultv_cu, itemCount, wi_cu);
        cudaMemcpy(resultv, resultv_cu, sizeof(float)*(BLOCK_NUM), cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+:val)
        for (int u = 0; u < BLOCK_NUM; u++){
          val += resultv[u];
        } 
        //cout<<val<<endl;
        res_cuda[f][k] = val;
        res_cuda[k][f] = val;
      }
    }
    //cout<<"1023 is okay"<<endl;
    cudaFree(V_clone_cu);
    cudaFree(V_cu);
    cudaFree(resultv_cu);
    cudaFree(wi_cu);
    free(V_h);
    free(V_clone_h);
    free(resultv);
    
    for(int i=0; i<factors; i++){
      for(int j=0; j<factors; j++){
        if(res_cpp[i][j]!=res_cuda[i][j])
          std::cout<<"i: "<<i<<" j: "<<j<<" cpp: "<<res_cpp[i][j]<<" cuda: "<<res_cuda[i][j]<<endl;
      }
    }
    
}

int main(){
  //updateUserSchedule1();
  //testUpdateSu();
  testUpdateSv();
  return 0;
}
