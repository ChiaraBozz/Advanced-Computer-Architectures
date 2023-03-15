#include <cstdio>   // std::printf
#include <iostream>
#include <string>
#include <omp.h>    // openmp header

int main() {
    //PARALLEL
    #pragma omp parallel num_threads(3)
    {
        printf("Hello!! I'm thread n.%i\n", omp_get_thread_num());
    }
    printf("\n\n\n");
    
    
    
    
    //FOR clause
    #pragma omp parallel for ordered
    for (int n = 0; n < 10; ++n){
        #pragma omp ordered //: check with and without this clause
        printf("iteration %d (I'm thread n.%i)\n", n, omp_get_thread_num());
    }
    printf("\n\n\n");
    
    
    
    //Private/shared variables
    int a=5, b;
    printf("I'm thread master n.%i. a=%i\n", omp_get_thread_num(), a);
    #pragma omp parallel private(a, b)
    {
        printf("I'm thread n.%i. a=%i\n", omp_get_thread_num(), a);
    }
    printf("\n\n\n");
    
    
    
    
    //firstprivate
    printf("First private\n");
    #pragma omp parallel firstprivate(a)
    {
        printf("I'm thread n.%i. a=%i\n", omp_get_thread_num(), a);
        printf("I'm thread n.%i. b=%i\n", omp_get_thread_num(), b);
    }
    
    printf("\n\n\n");
    

     
    
    double start;
    double end;
    start = omp_get_wtime();
    
    
    
    
    printf("Last private\n");
    #pragma omp parallel for lastprivate(a, b)
    for (int n = 0; n < 10; ++n){
            //#pragma omp ordered : check with and without this clause
            printf("I'm thread n.%i. a=%i\n", omp_get_thread_num(),a);
            a=omp_get_thread_num();
        }
    
    
    printf("I'm thread master n.%i. a=%i\n", omp_get_thread_num(), a);
    printf("\n\n\n");
    
    
    
    end = omp_get_wtime();
    printf("Time spent by the for loop: %f seconds\n", end - start);
    
    
    
    
    
    
    return 0;
}
