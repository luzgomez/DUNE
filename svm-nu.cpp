#include <iostream>
#include <dlib/svm.h>
using namespace std;
using namespace dlib;
int main()
{   typedef matrix<double, 2, 1> matype;
  typedef radial_basis_kernel<matype> fkernel;// iniciar  clase  function radial kernel
    std::vector<matype> samples;
    std::vector<double> labels;

    // data
    for (int i = -20; i <= 20; ++i)
    {
        for (int j = -20; j <= 20; ++j)
        {  matype ejp;
            ejp(0) = i;
            ejp(1) = j;
            samples.push_back(ejp);

            // mayor y menor a 10
            if (sqrt((double)i*i + j*j) <= 10){labels.push_back(+1);}
            else{         labels.push_back(-1);} }   }
  
    vector_normalizer<matype> normalizer;
    normalizer.train(samples);

    for (unsigned long i = 0; i < samples.size(); ++i){
      samples[i] = normalizer(samples[i]); }

  
    randomize_samples(samples, labels);// poner de manera aleatoria los datos para
    // efectuar cross validation

// El parámetro nu tiene un valor máximo que depende de la relación de +1 a -1
     // etiquetas en los datos de entrenamiento. Esta función encuentra ese valor.
    const double max_nu = maximum_nu(labels);

    svm_nu_trainer<fkernel> entrenador; // declarar entrenador de svm con kernel rbf

    // grilla para buscar parametros gamma nu
    cout << "cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double nu = 0.00001; nu < max_nu; nu *= 5)
        {   entrenador.set_kernel(fkernel(gamma));
            entrenador.set_nu(nu);

            cout << "gamma: " << gamma << "    nu: " << nu;
            //Imprima la precisión de la validación cruzada para una validación cruzada de 3 veces usando
             // el gamma actual y nu. cross_validate_trainer () devuelve un vector de fila.
             // El primer elemento del vector es la fracción de +1 ejemplos de entrenamiento
             // correctamente clasificado y el segundo número es la fracción de -1 entrenamiento
             // Ejemplos correctamente clasificados.
            cout << "  cross validation accuracy: " << cross_validate_trainer(entrenador, samples, labels, 3);
        } }

  // De mirar la salida del loop anterior resulta que un buen valor para nu
     // y gamma para este problema es 0.15625 para ambos. Así que eso es lo que usaremos.

     // Ahora entrenamos en el conjunto completo de datos y obtenemos la función de decisión resultante. Nosotros
     // usa el valor de 0.15625 para nu y gamma. La función de decisión devolverá valores.
     //> = 0 para las muestras que predice están en la clase +1 y los números <0 para las muestras que
     // predice estar en la clase -1.
    entrenador.set_kernel(fkernel(0.15625));
    entrenador.set_nu(0.15625);
    typedef decision_function<fkernel> decisionfunct;
    typedef normalized_function<decisionfunct> functype;

// Aquí estamos haciendo una instancia del objeto de función normalizada. Este objeto
     // proporciona una manera conveniente de almacenar la información de normalización del vector junto con
     // la función de decisión que se va a aprender.
      functype learned_function;
    learned_function.normalizer = normalizer; 
    learned_function.function = entrenador.train(samples, labels); 
 
    cout << "El número de vectores de soporte en nuestra función de aprendizaje son " 
         << learned_function.function.basis_vectors.size() << endl;

    //ejemplos
    matype ejp;

    ejp(0) = 3.123;
    ejp(1) = 2;
    cout << "este es un ejemplo de clase +1, la salida del clasificador es " << learned_function(ejp) << endl;


}



