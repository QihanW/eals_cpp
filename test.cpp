#include<iostream>
#include<immintrin.h>

#define NUM_DOUBLE 8

using namespace std;

int main(){
  
  /* double *uget = new double[64];
  double *svget = new double[64];
  double *numer = new double[8];
  for (int i=0; i<64; i++){
    uget[i] = 1;
    svget[i] = 2;
   // numer[i] = 0;
  }
  //double numer = 0;
  int f = 34;
  //__m512d tmp_zero = _mm512_setzero_pd ();
  _mm512_store_pd(numer, _mm512_setzero_pd ());  
  for (int k=0; k<64; k+=NUM_DOUBLE){
    //__m512d zero = _mm512_setzero_pd ();

    //if(k != f)
    //  numer -= (*(uget+k)) * (*(svget+k)); 
    //
    __m512d uget_k = _mm512_load_pd(uget + k);
    __m512d svget_k = _mm512_load_pd(svget + k);
    __m512d num = _mm512_load_pd(numer);
    __m512d tmp_mul = _mm512_mul_pd(uget_k, svget_k);
    __m512d tmp_add = _mm512_add_pd(num, tmp_mul);
    _mm512_store_pd(numer, tmp_add);
  }
  double sum = 0;
  for (int k=0; k<8; k++){
      sum -= numer[k];
    //cout<<numer[k]<<endl;
  }
  cout<<sum<<endl;

  long long *itemList = new long long[64];
  double *prediction_items = new double[64];
  double *v_col = new double[64];
  double *pre1 = new double[64];
  double tmp_uget = 10;
  int j;
  int sum = 0;
for (int i=0; i<29; i++)
  itemList[i] = i*2;
for (int i=29; i<64; i++)
  itemList[i] = i-19;
  for (int i=0; i<64; i++){                                                     
       
        // itemList[i] = 64 - i;
       prediction_items[i] = 2;
       pre1[i] = 2;
       v_col[i] = 3;                                                             
   }
  
  for (int k=0; k<64; k++){
    j = itemList[k];
    pre1[j] += tmp_uget * v_col[k];
    if(k%8 == 7){
  //    cout<<pre1[k]<<endl;
    }
  }
  
  double * x = new double[8];
  for (int i=0;i<8;i++){
    x[i] = i;
  }
  

  
  for (int j=0; j<64; j+=NUM_DOUBLE){                                 
 __m512i i_avx = _mm512_load_epi64(itemList + j);                           
 __m512d vcol_avx = _mm512_load_pd(v_col + j);                           
 __m512d tmp_mul = _mm512_mul_pd(_mm512_set1_pd(tmp_uget), vcol_avx);    
__m512d pre_avx = _mm512_i64gather_pd(i_avx, prediction_items, sizeof(double));
 __m512d sum_avx = _mm512_add_pd(tmp_mul, pre_avx);                      
 _mm512_i64scatter_pd(prediction_items, i_avx, sum_avx, sizeof(double)); 
  }  
  

  for(int i=0; i<64; i++){
    if ( prediction_items[i] = pre1[i])
    //  cout<<prediction_items[i]<<endl;
     cout<<i<<" correct"<<endl;
    else
      cout<<i<<" wrong"<<endl;
  }
  
  //cout<<sum<<endl;

  delete [] itemList;
  delete [] prediction_items;
  delete [] v_col;
  delete [] pre1;
  delete [] x; */



  double a[64];
  double b[64];
  double mius = -1;
  for (int i=0; i<64; i++){
    a[i] = 1;
    b[i] = 2;
  }
  for (int j=0; j<64; j+=NUM_DOUBLE){
    __m512d uget_k = _mm512_load_pd(a + j);
    __m512d min = _mm512_set1_pd(mius);
    __m512d tmp_mul = _mm512_mul_pd(min, uget_k);
    _mm512_store_pd(b + j, tmp_mul);
  }

  for(int i=0; i<64; i++)
    cout<<b[i]<<endl;






}
