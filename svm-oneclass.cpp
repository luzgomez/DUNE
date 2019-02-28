#include <iostream>
#include <vector>
#include <dlib/svm.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>
#include <dlib/array2d.h>
using namespace std;
using namespace dlib;

//funcion a la que se tiene que parecer svm
double sinc(double x)
{
    if (x == 0)
        return 2;
    return 2*sin(x)/x;}

int main()
{    typedef matrix<double,0,1> matype;// tipo de matriz a usar todo el codigo

  typedef radial_basis_kernel<matype> fkernel;//inicializar funcion kernel

  
  svm_one_class_trainer<fkernel> trainer;//tipo de svm a usar
// Aquí establecemos el ancho del núcleo de base radial a 4.0. Los valores más grandes hacen que
     // Ancho más pequeño y le dan al núcleo de base radial una mayor resolución. Si juegas con
     // el valor y observe la salida del programa obtendrá una sensación más intuitiva de lo que
     // eso significa.
    trainer.set_kernel(fkernel(4.0));

    //datos de entrenamiento
    std::vector<matype> samples;
    matype m(2);
    for (double x = -15; x <= 8; x += 0.3)
    {
        m(0) = x;
        m(1) = sinc(x);
        samples.push_back(m);}

   
// Ahora entrena un SVM de una clase. El resultado es una función, df (), que genera una salida grande.
     // valores para puntos de la curva sinc () y valores más pequeños para puntos que son
     // anómalo (es decir, no en la curva sinc () en nuestro caso).


    decision_function<fkernel> df = trainer.train(samples);
  
    cout << "Puntos que SIS están en la función sinc.:\n";
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0;   m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -4.1; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << df(m) << endl;  

    cout << endl;

    cout << "Puntos que NO están en la función sinc.:\n";
    m(0) = -1.5; m(1) = sinc(m(0))+4;   cout << "   " << df(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+3;   cout << "   " << df(m) << endl;
    m(0) = -0;   m(1) = -sinc(m(0));    cout << "   " << df(m) << endl;
    m(0) = -0.5; m(1) = -sinc(m(0));    cout << "   " << df(m) << endl;
    m(0) = -4.1; m(1) = sinc(m(0))+2;   cout << "   " << df(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+0.9; cout << "   " << df(m) << endl;
    m(0) = -0.5; m(1) = sinc(m(0))+1;   cout << "   " << df(m) << endl;}





