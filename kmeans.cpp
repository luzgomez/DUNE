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
#include <chrono>
#include <random>
using namespace std;
vector<vector<double> > data;// matriz de datos
void show ()
{
  int z,f ; z=-1;;
    ifstream archivo("inputkmean1.txt");
    string linea;
    while (getline(archivo,linea))
    {
      f=0;
        data.push_back(vector<double>());
        istringstream mm(linea);
        double value;
	if (f==0){
	z++;
        while (mm >> value)
        {
            data[z].push_back(value);
	    f=1;
        }}  }}
//--------------------------------------------------
//distancia euclidea

double dist ( vector<double>& a, vector<double>& b){
  double dis;
  int n=2;// numero de dimensiones del espacio------------------->
  vector<double> c(n,0) ;
  std::transform (a.begin(), a.end(), b.begin(), c.begin(), std::minus<double>());
  dis= inner_product(c.begin(), c.end(), c.begin(), 0.0);
  return sqrt(dis);}

// maximo valor de una columna

double maxcol(vector<vector<double> >& p, int c){
  double max; max=p[0][c]; 
  for (int i = 0; i < p.size(); i++){
    //cout << p[i].size()<< endl;
    //cout<< p[i][c]<< endl;
    if (max<p[i][c]){max=p[i][c]; }    }
  return max; }
//minimo valor de una columna

double mincol(vector<vector<double> >& p, int c){
  double min; min=p[0][c]; 
  for (int i = 0; i < p.size(); i++){
    if (min>p[i][c]){min=p[i][c];}    }
  return min; }

// maximo valor de un vector

double maxvect(vector<double>& p){
  double max; max=p[0]; int j; j=0;
  for (int i = 0; i < p.size(); i++){
    if (max<p[i]){max=p[i]; }    }
  return max; }

void kmedias(vector<vector<double> >& MED,vector<vector<double> >& MED1,vector<vector<double> >& data, int numclust, double tol){
  double tol1= 10;// cambio maximo entre la medias despues de promediar
  while(tol<=tol1){
    //proceso de asignacion de cluster
    
  vector<int> cluster(data.size(),0);
  double medida, medida1;
  for (int i = 0; i < data.size(); i++){
    //cout<<"punto " <<i<<"------------"<<endl;
     medida=dist(MED[0],data[i]);
     //cout<<medida<<endl;     
    for (int j = 1; j < numclust; j++){   
      medida1=dist(MED[j],data[i]);
      //cout<<medida1<<endl;     
      if(medida1<=medida){
	//cout<<" if"<<endl;
	medida=dist(MED[j],data[i]);
	//cout<<medida<<endl;
	cluster[i]=j; }
      
	  }}
   cout<<"/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// "<< endl;
  cout<<"centroides "<< endl;
 for (int c = 0; c < numclust; c++){
    cout<<"("<<MED[c][0]<<", "<<MED[c][1]<<")"<<" centro "<< c <<endl;
 }
 cout<<"clusters"<<endl;
 for (int i = 0; i < data.size(); i++){
   // cout<<"("<<data[i][0]<<", "<<data[i][1]<<")"<<" pertenece al cluster "<< cluster[i] <<" punto " <<i<<"------------"<<endl;
 }
 //cout<<"-_________________________________________________________________________________________"<<endl;
   
  
 //actualizacion de medias mediante promedio de puntos en cada cluster
  for (int c = 0; c < numclust; c++){
        cout<<" "<<endl;
	cout<<"cluster"<< c<< endl;
  MED[c][0]=0; MED[c][1]=0;
  int s; s=0;
  for (int i = 0; i < cluster.size(); i++){
        if(cluster[i]==c){
	  cout<< " "<<i;
      MED[c][0]=MED[c][0]+data[i][0] ; MED[c][1]= MED[c][1]+data[i][1] ; s++;} }
  
   MED[c][0]=MED[c][0]/s; MED[c][1]= MED[c][1]/s;
    cout<<" "<<endl;
 ;}

   //---------------------------------------------------------------------
  //diferencia entre medias anteriores y actualizada
  cout<<" "<<endl;
  vector< double>  difcen;
  for (int c = 0; c < numclust; c++){
    difcen.push_back(dist(MED[c],MED1[c]));}
  tol1=maxvect(difcen);
      cout<<" "<<endl;
  cout<<"maximo movimiento entre centros "<<tol1<<endl;
  difcen.clear();
  //---------------------------------------------------------------------
  for (int c = 0; c < MED.size(); c++){
  MED1[c][0]=MED[c][0]; MED1[c][1]= MED[c][1];}
   
   }}


      
int main()
{  show();//recolecta de datos
  vector<vector<double> > MED, MED1;// media nueva y vieja
  double xmax, ymax, xmin, ymin;// delimitacion de rango limites de los datos
  int numclust; numclust=5;// numero de clusters
 
  xmax=maxcol(data,0); xmin=mincol(data,0); ymin=mincol(data,1); ymax=maxcol(data,1);

  // generacion aleatoria de medias
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::uniform_real_distribution<double> distributionx (xmin, xmax);
  std::uniform_real_distribution<double> distributiony (ymin, ymax);
  vector<double>pack;
  for (int i=0; i<numclust; ++i){
    pack.push_back(distributionx(generator));
    pack.push_back(distributiony(generator));
    MED.push_back(pack);
    MED1.push_back(pack);
    pack.clear();}
  double tol=0.1;
  
  kmedias(MED, MED1,  data,  numclust, tol);
   
    return 0;}
      

  
      

        
        
 
    
        
        
    
