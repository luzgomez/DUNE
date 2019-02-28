#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <math.h>
#include <iomanip>
using namespace std;

// distancia euclidea
double dist ( vector<double>& a, vector<double>& b){
  double dis;
  int n=2;
  vector<double> c(n,0) ;
   std::transform (a.begin(), a.end(), b.begin(), c.begin(), std::minus<double>());
  dis= inner_product(c.begin(), c.end(), c.begin(), 0.0);
  return sqrt(dis);}
//maximo elemento de un vector
double maxvect(vector<double>& p){
  double max; max=p[0]; int j; j=0;
  for (int i = 0; i < p.size(); i++){
    if (max<p[i]){max=p[i]; j=i;}    }
  return j; }
// knn encontrar vecino mas cercano
void nachbarn(vector<vector<double> >& H,vector<double>& He,vector<double>& p,int k){
   vector<double> d;  vector<double> l ;  double it;
     for (int i = 0; i < H.size(); i++)
       {//cout<< H[i][1]<<endl;
	 it=dist(H[i],p);
	 d.push_back(it);}
    it=0; int j; 
    while (it<k){
      j= maxvect(d);
      l.push_back(He[j]);
      He.erase(He.begin()+j);
      d.erase(d.begin()+j);
      it=it+1;}
    for (int i = 0; i <k; i++){
        cout<< l[i]<< " esta "<< std::count (l.begin(), l.end(), l[i])<<" veces"<< endl; }}
      
int main()
{
  // input
  vector<double> labels {1,2,0};// tipo de clase
     vector<vector<double> > M;
     // coordenadas
     vector<double> vect{ 1, 1.1 }; 
     M.push_back(vect);
     vect.erase(vect.begin(),vect.begin()+2);
     vect={ 1,0 };
     M.push_back(vect);
     vect.erase(vect.begin(), vect.begin()+2);
     vect={0, 0};
     M.push_back(vect);
     vect.erase(vect.begin(), vect.begin()+2);
     vector<double> p {1.1, 1.1};
     nachbarn(M,labels,p,3);
    return 0;}
      

  
      

        
        
 
    
        
        
    
