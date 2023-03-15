#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

__global__
void vectorAddKernel(const int* d_inputA,
                     const int* d_inputB,
                     int        N,
                     int*       output) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_id < N) output[global_id] = d_inputA[global_id] + d_inputB[global_id];
}

const int N = 100000000;

int main(){
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_inputA     = new int[N];
    int* h_inputB     = new int[N];
    int* d_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N];

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++) {
        h_inputA[i] = distribution(generator);
        h_inputB[i] = distribution(generator);
    }

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    std::cout<<"Starting computation on HOST.."<<std::endl;
    TM_host.start();

    for (int i = 0; i < N; i++)
        h_output[i] = h_inputA[i] + h_inputB[i];

    TM_host.stop();
    TM_host.print("vectorAdd host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_inputA, *d_inputB, *d_output;
    SAFE_CALL( cudaMalloc( &d_inputA, N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_inputB, N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_output, N * sizeof(int) ));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_inputA, h_inputA, N * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL( cudaMemcpy( d_inputB, h_inputB, N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // DEVICE INIT
    dim3 DimGrid(N/256, 1, 1);
    if (N%256) DimGrid.x++;
    dim3 DimBlock(256, 1, 1);
              
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    std::cout<<"Starting computation on DEVICE.."<<std::endl;
    TM_device.start();

    vectorAddKernel<<<DimGrid,DimBlock>>>(d_inputA,d_inputB,N,d_output);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("vectorAdd device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( d_output_tmp, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != d_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << d_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
//	else printf("%i %i\n", h_output[i], d_output_tmp[i]);
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_inputA;
    delete[] h_inputB;
    delete[] h_output;
    delete[] d_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_inputA ) );
    SAFE_CALL( cudaFree( d_inputB ) );
    SAFE_CALL( cudaFree( d_output ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
