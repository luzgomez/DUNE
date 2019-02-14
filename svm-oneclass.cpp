

#include <iostream>
#include <vector>
#include <dlib/svm.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>
#include <dlib/array2d.h>


using namespace std;
using namespace dlib;

//funcion a aprender
double sinc(double x)
{
    if (x == 0)
        return 2;
    return 2*sin(x)/x;
}

int main()
{

    typedef matrix<double,0,1> sample_type;

    typedef radial_basis_kernel<sample_type> kernel_type;

  .
    svm_one_class_trainer<kernel_type> trainer;

    trainer.set_kernel(kernel_type(4.0));


    std::vector<sample_type> samples;
    sample_type m(2);
    for (double x = -15; x <= 8; x += 0.3)
    {
        m(0) = x;
        m(1) = sinc(x);
        samples.push_back(m);
    }


    decision_function<kernel_type> df = trainer.train(samples);
  
    cout << "Puntos que SIS están en la función sinc.:\n";
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0;   m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -4.1; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  

    cout << endl;
r. 
    cout << "Puntos que NO están en la función sinc.:\n";
    m(0) = -1.5; m(1) = sinc(m(0))+4;   cout << "   " << df(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+3;   cout << "   " << df(m) << endl;
    m(0) = -0;   m(1) = -sinc(m(0));    cout << "   " << df(m) << endl
    m(0) = -0.5; m(1) = -sinc(m(0));    cout << "   " << df(m) << endl;
    m(0) = -4.1; m(1) = sinc(m(0))+2;   cout << "   " << df(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+0.9; cout << "   " << df(m) << endl;
    m(0) = -0.5; m(1) = sinc(m(0))+1;   cout << "   " << df(m) << endl;
// La salida es la siguiente:
        /*
     Puntos que están en la función sinc:
         0.000389691
         0.000389691
         -0.000239037
         -0.000179978
         -0.000178491
         0.000389691
         -0.000179978

    // Puntos que NO están en la función sinc:
         -0.269389
         -0.269389
         -0.269389
         -0.269389
         -0.269389
         -0.239954
         -0.264318
     * /




