#include <iostream>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;


int main()
{
    
    typedef matrix<double, 2, 1> matype;
    typedef radial_basis_kernel<matype> kerneltype;
    std::vector<matype> samples;
    std::vector<double> labels;
    // data
    for (int i = -20; i <= 20; ++i)
    {for (int j = -20; j <= 20; ++j)
        {matype samp;
            samp(0) =i; samp(1) = j;
            samples.push_back(samp);
            // mayor y menor que 10
            if (sqrt((double)i*i + j*j) <= 10){
	      labels.push_back(+1);}
            else{labels.push_back(-1);        } }
      
    // normalizar datos
    vector_normalizer<matype> normalizador;
    // obtiene media y desviacion estandar
    normalizador.train(samples);
    // 
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizador(samples[i]); 


    // existen los parametros gamma y C
    // se entrenara mediante validacion cruzada y para ello se debe poner de manera aleatoria los datos
    randomize_samples(samples, labels);

    svm_c_trainer<kerneltype> entrenador;// iniciar clase train svmc

    // se itera la grilla de posibilidades.
    cout << "realizando cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double C = 1; C < 100000; C *= 5)
        {
             
            entrenador.set_kernel(kerneltype(gamma));
            entrenador.set_c(C);

            cout << "gamma: " << gamma << "    C: " << C;
            // Imprime la precisión de la validación cruzada para una validación cruzada de 3 veces usando
             // la gama actual y C. cross_validate_trainer () devuelve un vector fila.
             // El primer elemento del vector es la fracción de +1 ejemplos de entrenamiento
             // correctamente clasificado y el segundo número es la fracción de -1 entrenamiento
             // Ejemplos correctamente clasificados.
            cout << " precision    cross validation : " 
                 << cross_validate_trainer(entrenador, samples, labels, 3); }}

     // los valores para C y gamma para este problema son 78125 y 0.03125 respectivamente.
    entrenador.set_kernel(kerneltype(0.03125));
    entrenador.set_c(78125);
    typedef decision_function<kerneltype> decisionfunct;
    typedef normalized_function<decisionfunct> funct_type;

    funct_type learned_function;
    learned_function.normalizer = normalizador;  // salvar la info
    learned_function.function = entrenador.train(samples, labels);
    // ejecute el entrenamiento real de SVM y guarde los resultados.

    //imprimir el número de vectores de soporte en la función de decisión resultante
    cout << "El número de vectores de soporte en nuestra función de aprendizaje es " 
         << learned_function.function.basis_vectors.size() << endl;

    // ejemplos.
    matype ejp;

    ejp(0) = 3.123;
    ejp(1) = 2;
    cout << "Este es un ejemplo de clase +1, la salida del clasificador es " << learned_function(ejp) << endl;

    ejp(0) = 13.123;
    ejp(1) = 0;
    cout << "Este es un ejemplo de clase -1, la salida del clasificador ess " << learned_function(ejp) << endl;}
}



