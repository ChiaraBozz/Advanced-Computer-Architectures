#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    int N = (1 << 25);

    int* Array = new int[N];
    const int to_find1 = 18;        //search [ .. ][ .. ][ 18 ][ 64 ][ .. ]
    const int to_find2 = 64;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 10000);

    for (int i = 0; i < N; ++i)
        Array[i] = distribution(generator);

    // ------------ SEQUENTIAL SEARCH ------------------------------------------
    Timer<HOST> TM;
    TM.start();

    int index = -1;
    for (int i = 0; i < N - 1; ++i) {
        if (Array[i] == to_find1 && Array[i + 1] == to_find2) {
            index = i;
            break;            // !!! error in OPENMP
        }
    }

    TM.stop();
    TM.print("Sequential Search");
    std::cout << index << std::endl;

    // ------------ PARALLEL SEARCH --------------------------------------------
    TM.start();

    index = -1;
    /*
     
     TO COMPLETE
     */

    TM.stop();
    TM.print("Parallel Search");
    std::cout << index << std::endl;
}
