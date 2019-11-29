#include <iostream>
using namespace std;

#define BLOCK_NUM 64
#define THREAD_NUM 256

__global__ void test(float **data1, float **data2, float *res){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  //__shared__ float numer[THREAD_NUM];;
  float numer;
  for(int u = bid*THREAD_NUM+tid; u < 20; u+=BLOCK_NUM*THREAD_NUM){
    for (int f = 0; f < 10; f++) {
      numer = 0;
      for(int k = 0; k<10; k++){
        numer -= data1[u][k] * data2[f][k];
       // numer[u] -= f*k;
        
      }
      //res[u] = numer[u];
     // printf("%f ", numer);
    }
    res[u] = numer;
  }
}
int main(){
  
  float **data1, **data2;
  data1 = (float **)malloc(sizeof(float *)*20);
  data2 = (float **)malloc(sizeof(float *)*10);
  for(int i=0; i<20; i++){
    data1[i] = (float *)malloc(sizeof(float)*10);
    //data2[i] = (float *)malloc(sizeof(float)*10);
    for(int j=0; j<10; j++){
      data1[i][j] = (float)i+j;
    }
  }
  for(int i=0; i<10; i++){
    data2[i] = (float *)malloc(sizeof(float)*10);
    for(int j=0; j<10; j++){
      data2[i][j] = (float)i-j;
    }
  }
  
  float numer;
  /*
  for(int u=0; u<20; u++){
    for(int f=0; f<10; f++){
      numer = 0;
      for(int k = 0; k<10; k++){
        //numer -= data1[u][k] * data2[f][k];
        numer -= f*k;
      }
      //cout<<numer<<" ";
      //

    }
  }
  */

  float **data_cu1, **data_cu2;
  cudaMalloc((void**)&data_cu1,sizeof(float *)*20);
  cudaMalloc((void**)&data_cu2,sizeof(float *)*10);
  float **copy1, **copy2;
  copy1 = (float **)malloc(sizeof(float *)*20);
  copy2 = (float **)malloc(sizeof(float *)*10);    
  for(int i=0; i<20; i++){
    float *tmp;
    cudaMalloc((void**)&tmp,sizeof(float)*10); 
    cudaMemcpy(tmp, data1[i], sizeof(float)*10, cudaMemcpyHostToDevice);
    //cudaMemcpy(tmp, data2[i], sizeof(float)*10, cudaMemcpyHostToDevice);
    copy1[i] = tmp;
  }
  for(int i=0; i<10; i++){                                                      
     float *tmp;                                                                 
     cudaMalloc((void**)&tmp,sizeof(float)*10);                                
    // cudaMemcpy(tmp, data1[i], sizeof(float)*10, cudaMemcpyHostToDevice);        
     cudaMemcpy(tmp, data2[i], sizeof(float)*10, cudaMemcpyHostToDevice);      
     copy2[i] = tmp;                                                             
  }    

  cudaMemcpy(data_cu1, copy1, sizeof(float *)*20, cudaMemcpyHostToDevice); 
  cudaMemcpy(data_cu2, copy2, sizeof(float *)*10, cudaMemcpyHostToDevice); 

  float *res_cu;
  cudaMalloc((void**)&res_cu,sizeof(float)*20);
  //test<<<BLOCK_NUM,THREAD_NUM>>>(data_cu1, data_cu2, res_cu);
  float *res;
  res = (float *)malloc(sizeof(float)*20);
  test<<<BLOCK_NUM,THREAD_NUM>>>(data_cu1, data_cu2, res_cu);
  cudaMemcpy(res, res_cu, sizeof(float)*20, cudaMemcpyDeviceToHost);
  for(int u=0; u<20; u++){                                                      
    for(int f=0; f<10; f++){                                                    
      numer = 0;                                                                
      for(int k = 0; k<10; k++){                                                
        //numer -= f*k; 
        numer -= data1[u][k] * data2[f][k];
      }
    //if(numer != res[u])
      //cout<<"wrong! "<<"numer: "<<numer<<" res_cu: "<<res_cu[u]<<endl;
    }                    
    if(numer != res[u])
      cout<<"wrong! "<<"numer: "<<numer<<" res_cu: "<<res_cu[u]<<endl;
  }   

  
  cudaFree(res_cu);
  free(res);

  for(int u=0; u<20; u++)
  {  
    free(data1[u]);
    cudaFree(copy1[u]);
  }
  for(int u=0; u<10; u++){
    free(data2[u]);
    cudaFree(copy2[u]);
  }
  free(data1);
  free(data2);
  free(copy1);
  free(copy2);
  cudaFree(data_cu1);
  cudaFree(data_cu2);
  
  cout<<endl;

}
