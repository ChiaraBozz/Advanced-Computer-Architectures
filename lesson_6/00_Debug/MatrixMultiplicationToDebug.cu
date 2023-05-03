#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

#define TILE 16
const int N = 512;
const int BLOCK_SIZE_X = TILE;
const int BLOCK_SIZE_Y = TILE;

__global__
void matrixMultiplicationKernel(int* d_M, int* d_N, int Width, int* d_P)
{
    __shared__ int ds_M[TILE][TILE];
    __shared__ int ds_N[TILE][TILE];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // Identify the row and column of the Pd element to work on
    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;
    int Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < Width/TILE; ++m) {
        // Coolaborative loading of Md and Nd tiles into shared memory
        ds_M[ty][tx] = d_M[Row*N + m*TILE+tx];
        ds_N[ty][tx] = d_N[Col+(m*TILE+ty)*N];
        if (tx == 0) d_P[N*N] = 0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        __syncthreads();
    }
    d_P[Row*N+Col] = Pvalue;
}


int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_matrixA    = new int[N * N];
    int* h_matrixB    = new int[N * N];
    int* h_matrix_tmp = new int[N * N]; // <-- used for device result
    int* h_matrixC    = new int[N * N];

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N * N; i++) {
        h_matrixA[i] = distribution(generator);
        h_matrixB[i] = distribution(generator);
    }
    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    std::cout<<"Starting computation on HOST.."<<std::endl;
    TM_host.start();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                 sum += h_matrixA[i * N + k] * h_matrixB[k * N + j];
            h_matrixC[i * N + j] = sum;
        }
    }

    TM_host.stop();
    TM_host.print("MatrixMultiplication host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_matrixA, *d_matrixB, *d_matrixC;
    SAFE_CALL( cudaMalloc( &d_matrixA, N*N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_matrixB, N*N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_matrixC, N*N * sizeof(int) ));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_matrixA, h_matrixA, N*N * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL( cudaMemcpy( d_matrixB, h_matrixB, N*N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // DEVICE INIT
    dim3 DimGrid(N/BLOCK_SIZE_X, N/BLOCK_SIZE_Y, 1);
    if (N%BLOCK_SIZE_X) DimGrid.x++;
    if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();
    matrixMultiplicationKernel<<< DimGrid,DimBlock>>> (d_matrixA, d_matrixB, N, d_matrixC);

    CHECK_CUDA_ERROR
    TM_device.stop();
    TM_device.print("MatrixMultiplication device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_matrix_tmp, d_matrixC, N*N * sizeof(int), cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N * N; i++) {
        if (h_matrixC[i] != h_matrix_tmp[i]) {
            std::cerr << "wrong result at: ("
                      << (i / N) << ", " << (i % N) << ")"
                      << "\nhost:   " << h_matrixC[i]
                      << "\ndevice: " << h_matrix_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_matrixC;
    delete[] h_matrix_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_matrixA ));
    SAFE_CALL( cudaFree( d_matrixC ));
    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
