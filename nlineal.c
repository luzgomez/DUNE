// regresion multilineal cuadratica
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multifit.h>

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
 
double y[] = {	52.21, 53.12, 54.48, 55.84, 57.20,
		58.57, 59.93, 61.29, 63.11, 64.47,
		66.28, 68.10, 69.92, 72.19, 74.46 };// coordenadas en y
double x[] = {	1.47, 1.50, 1.52, 1.55, 1.57,
		1.60, 1.63, 1.65, 1.68, 1.70,
		1.73, 1.75, 1.78, 1.80, 1.83	};// coordenadas en x
 
int main()
{
	int n = sizeof(x)/sizeof(double);// forma de hallar el tamaño de un vector en C
	gsl_matrix *X = gsl_matrix_calloc(n, 3);// inicia la matriz de ceros
	gsl_vector *Y = gsl_vector_alloc(n);//inicializa un vector, le separa memoria
	gsl_vector *coef = gsl_vector_alloc(3);// vector de coef de polinimio cuadratico
 
	for (int i = 0; i < n; i++) {
		gsl_vector_set(Y, i, y[i]);
 
		gsl_matrix_set(X, i, 0, 1);// rellenar la matriz de punto X de forma cuadratica
		gsl_matrix_set(X, i, 1, x[i]);
		gsl_matrix_set(X, i, 2, x[i] * x[i]);
	}
 
	double chisq;// chi estadistico
	gsl_matrix *cov = gsl_matrix_alloc(3, 3);// matriz de covarianza
	gsl_multifit_linear_workspace * wspc = gsl_multifit_linear_alloc(n, 3);// inicializacionde regresion multilineal
	gsl_multifit_linear(X, Y, coef, cov, &chisq, wspc);// ejecucion de regresion
	
 
	printf("coeficientes:");
	for (int i = 0; i < 3; i++)
		printf("  %g", gsl_vector_get(coef, i));
	printf("\n");
 
	gsl_matrix_free(X);
	gsl_matrix_free(cov);//liberacion de espacio de memoria

	gsl_vector_free(Y);
	gsl_vector_free(coef);
	gsl_multifit_linear_free(wspc);
 
}


