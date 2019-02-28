#include <iostream>
#include <vector>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;

// funcion a modelar
// object.
double cosenc(double x)
{
    if (x == 0)
        return 1;
    return cos(x)*2;
}

int main()
{
     
  typedef matrix<double,1,1> vectype;// tipo de matrix a usar en el codigo
  typedef radial_basis_kernel<vectype> fkernel;// tipo de funcion kernel 

  // vectores para guardar datos de entrenamiento
    std::vector<vectype> ejpx;
    std::vector<double> ejpy;

    
    vectype m;
    for (double x = -10; x <= 4; x += 1)
    {
        m(0) = x;

        ejpx.push_back(m);
        ejpy.push_back(cosenc(x));    }

    // 3 parametros, kernel y 
    // 2 parametros especificos de SVR.  
    svr_trainer<fkernel> trainer;// inicializar entrenador  de datos
    trainer.set_kernel(fkernel(0.1));

    // PARMATETRO c
    trainer.set_c(10);

    // VALOR DE TOLERANCIA
    trainer.set_epsilon_insensitivity(0.001);

    // REGRESSION, se entrena la funcion kernel
    decision_function<fkernel> df = trainer.train(ejpx, ejpy);

    // PREDICCION
    m(0) = 2.5; cout << cosenc(m(0)) << "   " << df(m) << endl;
    m(0) = 0.1; cout << cosenc(m(0)) << "   " << df(m) << endl;
    m(0) = -4;  cout << cosenc(m(0)) << "   " << df(m) << endl;
    m(0) = 5.0; cout << cosenc(m(0)) << "   " << df(m) << endl;

 
   

    // 5-FOLD VALIDATION   
    randomize_samples(ejpx, ejpy);// desorganizar base de datos para cross validation
    cout << "MSE and R-Squared: "<< cross_validate_regression_trainer(trainer, ejpx, ejpy, 8) << endl;
    // output 
    // MSE y R-Squared: 1.65984e-05    0.999901
}




