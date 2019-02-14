


#include <iostream>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;


int main()
{
    
    typedef matrix<double, 2, 1> sample_type;

    typedef radial_basis_kernel<sample_type> kernel_type;

    std::vector<sample_type> samples;
    std::vector<double> labels;

    // data
    for (int r = -20; r <= 20; ++r)
    {
        for (int c = -20; c <= 20; ++c)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            // mayor y menor que 10
            if (sqrt((double)r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);

        }
    }


    // normalizar datos
    vector_normalizer<sample_type> normalizer;
    // obtiene media y desviacion estandar
    normalizer.train(samples);
    // 
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    // existen los parametros gamma y C
    // se entrenara mediante validacion cruzada y para ello se debe poner de manera aleatoria los datos
    randomize_samples(samples, labels);


    svm_c_trainer<kernel_type> trainer;

    // se itera la grilla de posibilidades.
    cout << "realizando cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double C = 1; C < 100000; C *= 5)
        {
             
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_c(C);

            cout << "gamma: " << gamma << "    C: " << C;
            // Imprime la precisión de la validación cruzada para una validación cruzada de 3 veces usando
             // la gama actual y C. cross_validate_trainer () devuelve un vector fila.
             // El primer elemento del vector es la fracción de +1 ejemplos de entrenamiento
             // correctamente clasificado y el segundo número es la fracción de -1 entrenamiento
             // Ejemplos correctamente clasificados.
            cout << " precision    cross validation : " 
                 << cross_validate_trainer(trainer, samples, labels, 3);
        }
    }

     // los valores para C y gamma para este problema son 5 y 0.15625 respectivamente.
   


    trainer.set_kernel(kernel_type(0.15625));
    trainer.set_c(5);
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;
.  
    funct_type learned_function;
    learned_function.normalizer = normalizer;  // salvar la info
    learned_function.function = trainer.train(samples, labels); // ejecute el entrenamiento real de SVM y guarde los resultados.

    //imprimir el número de vectores de soporte en la función de decisión resultante
    cout << "El número de vectores de soporte en nuestra función de aprendizaje es " 
         << learned_function.function.basis_vectors.size() << endl;

    // ejemplos.
    sample_type sample;

    sample(0) = 3.123;
    sample(1) = 2;
    cout << "Este es un ejemplo de clase +1, la salida del clasificador es " << learned_function(sample) << endl;

    sample(0) = 3.123;
    sample(1) = 9.3545;
    cout << "Este es un ejemplo de clase +1, la salida del clasificador es " << learned_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 9.3545;
    cout << "Este es un ejemplo de clase -1, la salida del clasificador es " << learned_function(sample) << endl;

    sample(0) = 13.123;
    sample(1) = 0;
    cout << "Este es un ejemplo de clase -1, la salida del clasificador ess " << learned_function(sample) << endl;




}



