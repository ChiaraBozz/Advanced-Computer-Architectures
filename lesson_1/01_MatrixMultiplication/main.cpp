#include <chrono>
#include <iostream>     // std::cout, std::fixed
#include <random>
#include "MatrixMultiplication.hpp"
#include "Timer.hpp"

int main() {
    using namespace timer;
    // -------------------- Matrix Allocation ----------------------------------
    /*  C++ 11 Style
    auto A = new int[ROWS][COLS];
    auto B = new int[ROWS][COLS];
    auto C = new int[ROWS][COLS];*/

    int** A = new int*[ROWS];
    int** B = new int*[ROWS];
    int** C_seq = new int*[ROWS];
    int** C_par = new int*[ROWS];
    for (int i = 0; i < ROWS; i++) {
        A[i] = new int[COLS];
        B[i] = new int[COLS];
        C_seq[i] = new int[COLS];
        C_par[i] = new int[COLS];
    }
    // -------------------- Matrix Filling -------------------------------------

    std::cout << std::endl << "Matrix Filling...";

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j)
            A[i][j] = distribution(generator);
    }
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j)
            B[i][j] = distribution(generator);
    }
    
    std::cout << "done!" << std::endl;

    // =========================================================================
    // ----------------- Matrix Multiplication Sequential ----------------------
    std::cout << std::endl << "Starting Sequential Mult....";
    Timer<HOST> TM;
    TM.start();

    sequentialMatrixMatrixMul(A, B, C_seq);

    TM.stop();
    std::cout << "done!" << std::endl<< std::endl;
    TM.print("Sequential Matrix-Matrix Multiplication");

    // ----------------- Matrix Multiplication OPENMP --------------------------
    std::cout << std::endl << "Starting Parallel Mult....";
    TM.start();

    openmpMatrixMatrixMul(A, B, C_par);

    TM.stop();
    std::cout << "done!" << std::endl<< std::endl;
    TM.print("OpenMP Matrix-Matrix Multiplication");

    
    // =========================================================================
    // ----------------- Check the results (C_seq = C_par) ----------------------
    
    int error=0;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j)
            if (C_seq[i][j] != C_par[i][j]){
                error=1;
                break;
            }
    }
    if(error)
        std::cout << std::endl <<"ERROR in Checking!" << std::endl<< std::endl;
    else
        std::cout << std::endl <<"Check OK!" << std::endl<< std::endl;
    
    
}
