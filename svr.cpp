#include <iostream>
#include <vector>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;

// funcion a modelar
// object.
double sinc(double x)
{
    if (x == 0)
        return 1;
    return sin(x)/x;
}

int main()
{
     
    typedef matrix<double,1,1> sample_type;

    
    typedef radial_basis_kernel<sample_type> kernel_type;


    std::vector<sample_type> samples;
    std::vector<double> targets;

    
    sample_type m;
    for (double x = -10; x <= 4; x += 1)
    {
        m(0) = x;

        samples.push_back(m);
        targets.push_back(sinc(x));
    }

    // 3 parametros, kernel y 
    // 2 parametros especificos de SVR.  
    svr_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(0.1));

    // PARMATETRO c
    trainer.set_c(10);

    // VALOR DE TOLERANCIA
    trainer.set_epsilon_insensitivity(0.001);

    // REGRESSION
    decision_function<kernel_type> df = trainer.train(samples, targets);

    // PREDICCION
    m(0) = 2.5; cout << sinc(m(0)) << "   " << df(m) << endl;
    m(0) = 0.1; cout << sinc(m(0)) << "   " << df(m) << endl;
    m(0) = -4;  cout << sinc(m(0)) << "   " << df(m) << endl;
    m(0) = 5.0; cout << sinc(m(0)) << "   " << df(m) << endl;

    // output 
    //  0.239389   0.23905
    //  0.998334   0.997331
    // -0.189201   -0.187636
    // -0.191785   -0.218924

     

    // 5-FOLD VALIDATION   
    randomize_samples(samples, targets);
    cout << "MSE and R-Squared: "<< cross_validate_regression_trainer(trainer, samples, targets, 5) << endl;
    // output 
    // MSE y R-Squared: 1.65984e-05    0.999901
}




