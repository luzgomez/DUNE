


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

            // mayor y menor a 10
            if (sqrt((double)r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);

        }
    }

  
    vector_normalizer<sample_type> normalizer;

    normalizer.train(samples);

    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    
    randomize_samples(samples, labels);


// El parámetro nu tiene un valor máximo que depende de la relación de +1 a -1
     // etiquetas en los datos de entrenamiento. Esta función encuentra ese valor.
    const double max_nu = maximum_nu(labels);

    svm_nu_trainer<kernel_type> trainer;

    // grilla
    cout << "cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double nu = 0.00001; nu < max_nu; nu *= 5)
        {
            // 
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_nu(nu);

            cout << "gamma: " << gamma << "    nu: " << nu;
            //Imprima la precisión de la validación cruzada para una validación cruzada de 3 veces usando
             // el gamma actual y nu. cross_validate_trainer () devuelve un vector de fila.
             // El primer elemento del vector es la fracción de +1 ejemplos de entrenamiento
             // correctamente clasificado y el segundo número es la fracción de -1 entrenamiento
             // Ejemplos correctamente clasificados.
            cout << "     cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);
        }
    }


  // De mirar la salida del loop anterior resulta que un buen valor para nu
     // y gamma para este problema es 0.15625 para ambos. Así que eso es lo que usaremos.

     // Ahora entrenamos en el conjunto completo de datos y obtenemos la función de decisión resultante. Nosotros
     // usa el valor de 0.15625 para nu y gamma. La función de decisión devolverá valores.
     //> = 0 para las muestras que predice están en la clase +1 y los números <0 para las muestras que
     // predice estar en la clase -1.
    trainer.set_kernel(kernel_type(0.15625));
    trainer.set_nu(0.15625);
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;

// Aquí estamos haciendo una instancia del objeto de función normalizada. Este objeto
     // proporciona una manera conveniente de almacenar la información de normalización del vector junto con
     // la función de decisión que vamos a aprender.
    funct_type learned_function;/ save normalization information
    learned_function.function = trainer.train(samples, labels); 

    
    cout << "El número de vectores de soporte en nuestra función de aprendizaje ess " 
         << learned_function.function.basis_vectors.size() << endl;

    //ejemplos
    sample_type sample;

    sample(0) = 3.123;
    sample(1) = 2;
    cout << "este es un ejemplo de clase +1, la salida del clasificador es " << learned_function(sample) << endl;

;
}



