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
vector<vector<double> > data;// vector de datos incial
vector<double> A;// vector x
vector<double> B;// vector y
//-----------------------------------------------------------------------------------------
//lectura de datos
//void show( string input)
void show ()
{
    //ifstream archivo(input);// file path
    ifstream archivo("inputc++1.txt");
    string linea;
    while (getline(archivo,linea))
    {
        data.push_back(vector<double>());
        istringstream mm(linea);
        double value;
        while (mm >> value)
        {
            data.back().push_back(value);
	    //std::cout << data[0].back() << '\n';
        }
    }
    for (int y = 0; y < data.size(); y++)
    {
        for (int x = 0; x < data[y].size(); x++)
        {
			if(x==0)
			{
				A.push_back(data[y][x]);
			//	std::cout << data[y][x] << '\n'; 		
			}
			else
			{
				B.push_back(data[y][x]);
			//	std::cout << data[y][x] << '\n';
			}       } }}
//-----------------------------------------------------------------------------------------
//minimos cuadrados obtencion de pendiente e intercepto
double slope(const std::vector<double>& x, const std::vector<double>& y)// minim
{
    const double n    = x.size();
    const double s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const double s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const double s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const double s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const double a    = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const double b    = ( s_y - s_x *a) / (n );
    cout << "slope" << "\n"; 
    return a; 
}
double intercept(const std::vector<double>& x, const std::vector<double>& y)
{
    const double n    = x.size();
    const double s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const double s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const double s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const double s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const double a    = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const double b    = ( s_y - s_x *a) / (n );
    cout << "intercept" << "\n"; 
    return b;    
}
//-----------------------------------------------------------------------------------------
//minimos cuadrados obtencion de pendiente e intercepto
std::vector<double> interslop(const std::vector<double>& x, const std::vector<double>& y)
{
    const double n    = x.size();
    const double s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const double s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const double s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const double s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const double a    = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const double b    = ( s_y - s_x *a) / (n );
    std::vector<double> slopinter;
    slopinter.push_back(b);
    slopinter.push_back(a);
    cout << "intercept" << "   slope" << "\n";
    return slopinter;    
}
//-----------------------------------------------------------------------------------------
// media aritmetica
double mean( std::vector<double>& x)
{
    const double n    = x.size();
    const double s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const double meanx= s_x/n;
    return meanx;    
}
//-----------------------------------------------------------------------------------------
// funcion lineal
double lineal2(std::vector<double>& param, double x)
{
  double f_i =param[0]+ param[1]*x;
        return f_i;    
}
//-----------------------------------------------------------------------------------------
double SCE ( std::vector<double>& x,  std::vector<double>& y, std::vector<double>& param){
  double fi=0;
  double sce=0;
  for (int i = 0; i < x.size(); i++){
    //   std::cout << fi <<'\n';
    fi=lineal2(param,x[i]);
    //  std::cout << fi <<'\n';
    fi= fi-y[i];
    //std::cout << fi <<'\n';
    sce=sce+ pow(fi,2);
    //std::cout << sce <<'\n';
      }
   cout << "Suma de cuadrados debido al error yi-fi, SCE=" <<  sce <<'\n';
        return sce;
}
//-----------------------------------------------------------------------------------------
double SCT ( std::vector<double>& y,int s){
  //error que no entendiiiii
  double sct=0;
  double sctmean=mean(y);
  for (int i = 0; i < y.size(); i++){
  double counter= y[i]-sctmean;
    //std::cout << counter <<'\n';
    sct=sct+ pow(counter,2);
    //std::cout << sct <<'\n';
      }
  if (s==0){   cout << "Suma de cuadrados total yi-ymean, SCT=" <<  sct <<'\n';}
        return sct;
}
//--------------------------------------------------------------------------------------------
double S2 ( std::vector<double>& y){
  double s2=SCT(y,1)/(y.size()-1);
        return s2;
}

//-----------------------------------------------------------------------------------------
double SCR ( std::vector<double>& x,  std::vector<double>& y, std::vector<double>& param){
  double fi=0;
  double scr=0;
  // cout << scr <<'\n';
  double scrmean=mean(y);
  for (int i = 0; i < x.size(); i++){
    //std::cout << " " <<'\n';
    //std::cout << x[i] <<'\n';
    fi=lineal2(param,x[i]);
    //std::cout << fi <<'\n';
    fi= fi-scrmean;
    //std::cout << fi <<'\n'; 
    scr=scr+ pow(fi,2);
    //std::cout << scr <<'\n';
    //cout <<pow(fi,2) <<"  "<< fi*fi<< '\n';
      }
   cout << "Suma de cuadrados debido a la regresion fi-ymean SCR =" << scr <<'\n';
   return scr;}
//----------------------------------------------------------------------------------------- 
double CME(double sce, int n){
  double cme=  sce/(n-2);
   cout<< " " << "\n";
   cout<< "Varianza del error del modelo"<< '\n';
  cout <<"Cuadrados medios del error SCE/(n-2)  CME= "<<cme << "\n";
   cout<< " " << "\n";
   cout<< "Raiz de varianza de error de modelo"<<'\n';
  cout<<"Error estandar de estimacion= "<< sqrt(cme)<<"\n";
  return cme;
}
//----------------------------------------------------------------------------------------- 
double R_squared(double sce, double sct, std::vector<double>& param){
  double rr= 1-( sce/sct);
  cout<<'\n';
  cout <<"R_squared"<<'\n';
  cout <<"Coeficiente de determinacion 1-(SCE/SCT)   R*R= "<<rr << "\n";
   cout<< " " << "\n";
  if (param[1]<0){cout << "indice de correlacion lineal= "<<"-"<<sqrt(rr)<<'\n'; }
  else{cout << "indice de correlacion lineal= "<<sqrt(rr)<<'\n'; }
  return rr;
  }
//----------------------------------------------------------------------------------------- 
double S2_muestral_intercept(double sce,vector<double>& x){
  double s2mint= (sce/(x.size()-2))*((1/x.size()) + (pow(mean(x),2)/((x.size()-1)*S2(x))));
  cout<<'\n';
  cout<< "varianza muestral de estimador del intercepto"<<'\n';
  cout<<"[SCE/(n-2)]*[(1/n)+(xmean²)/((n-1)*S²_x)) ]="<<s2mint<<'\n';
return s2mint;}
//----------------------------------------------------------------------------------------- 
double S2_muestral_slope(double sce,vector<double>& x){
  double s2msl= (sce/(x.size()-2))/(S2(x)*(x.size()-1));
  cout<<'\n';
  cout<< "varianza muestral de estimador de la pendiente"<<'\n';
  cout<<"SCE/[(n-1)*S²(x)*n-2]="<<s2msl<<'\n';
return s2msl;}

//----------------------------------------------------------------------------------------- 

void lin2ANOVA (double SSReg, double SST, double SSRes, int n){
   cout<< " " << "\n";
  cout<< "Tabla ANOVA" << "\n";
 cout<<"-------------------------------------------------------------------------------------------" << "\n";
 std::cout<<std::left << std::setw(15) << "Fuente Variacion |"
	  <<std::right <<std::setw(15)<< "  Suma Cuadrados |"
          <<std::right <<std::setw(15)<<" Grados libertad |"
	  <<std::right <<std::setw(18)<<" Cuadrados medios|"  
          <<std::right <<std::setw(18)<<"valor-F     |"
	  << "\n";
 cout<<"--------------------------------------------------------------------------------------------" << "\n";
 std::cout<<std::left << std::setw(15) << "Regresion "
	  <<std::right <<std::setw(15)<< SSReg 
	  <<std::right <<std::setw(15)<< 1
          <<std::right <<std::setw(18)<< SSReg
          <<std::right <<std::setw(18)<< SSReg*(n-2)/SSRes
<< "\n";
 cout<<"--------------------------------------------------------------------------------------------" << "\n";
std::cout<<std::left << std::setw(15) << "Errores "
	 <<std::right <<std::setw(15)<< SSRes
         <<std::right <<std::setw(18)<<n-2
	 <<std::right <<std::setw(18)<< SSRes/(n-2) 
<<"\n";
 cout<<"---------------------------------------------------------------------------------------------" << "\n";
std::cout<<std::left << std::setw(15) << "Total"
         <<std::right <<std::setw(15)<< SST
         <<std::right <<std::setw(18)<<n-1
	 <<std::right <<std::setw(18)<< " "       
	 <<  "\n";
 cout<<"----------------------------------------------------------------------------------------------" << "\n";
}


//-----------------------------------------------------------------------------------------------------------------------------
void lin2Summary (double SSReg, double SST, double SSRes, int n, double S2msl, std::vector<double>& param,double S2mint){
   cout<< " " << "\n";
   cout<< " La ecuacion lineal de la regresion es= "<<param[0]<<"+"<<param[1]<<"*x"<<'\n';
     cout<< " " << "\n";
  cout<< " Summary" << "\n";
 cout<<"-------------------------------------------------------------------------------------------" << "\n";
 std::cout<<std::left << std::setw(15) <<"Prediccion |"
	  <<std::right <<std::setw(10)<<"Coeficientes |"
          <<std::right <<std::setw(15)<<"Suma Cuadrados |"
	  <<std::right <<std::setw(18)<<"Valor-T      |"  
          <<std::right <<std::setw(10)<<"valor-P |"
	  << "\n";
 cout<<"--------------------------------------------------------------------------------------------" << "\n";
 std::cout<<std::left << std::setw(15) << "Intercepto "
	  <<std::right <<std::setw(10)<< param[0] 
	  <<std::right <<std::setw(15)<< S2mint
          <<std::right <<std::setw(18)<< param[0]/S2mint
          <<std::right <<std::setw(10)<< " "
<< "\n";
 cout<<"--------------------------------------------------------------------------------------------" << "\n";
std::cout<<std::left << std::setw(15) << "Pendiente "
	 <<std::right <<std::setw(10)<< param[1]
         <<std::right <<std::setw(15)<<  S2msl
	 <<std::right <<std::setw(18)<< param[1]/S2msl
         <<std::right <<std::setw(10)<<  " "
<<"\n";
 cout<<"---------------------------------------------------------------------------------------------" << "\n";}

//---------------------------------------------------------------------------------------------------------------


int main()
{	//string input;	
	//getline (cin, input);
	//show(input);
         show();	
	//std::cout << slope(A, B) << '\n'; 
	//std::cout << intercept(A, B) << '\n';
	cout<<"media de y" <<"   media de x"<< '\n';
	cout<< mean(B)<<"            " << mean(A)<< '\n';
        std::vector<double> paramt = interslop(A,B);
	for (int i = 0; i < paramt.size(); i++)
	{
	  std::cout << paramt[i]<<"            ";}
	 cout << "\n";
	 
	  double scer= SCE(A,B,paramt);
	  double sctr= SCT(B,0);
	  double scrr= SCR(A,B,paramt);
	  
	  double S2msl= S2_muestral_slope(scer,A);
	  double S2mint= S2_muestral_intercept(scer,A);
	  
	  lin2Summary(scrr,sctr,scer, A.size(),S2msl,paramt,S2mint);

	  double cme = CME(scer, A.size());
	  double RR = R_squared(scer,sctr, paramt);
	  
	  lin2ANOVA(scrr,sctr,scer, A.size());
      
      return 0;
  }

