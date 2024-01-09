/*
Prefix Sum for large arrays 
v1: naive
*/
#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

using namespace timer;
//using namespace timer_cuda;

#define DIV(a,b)	(((a) + (b) - 1) / (b))

//const int BLOCK_SIZE = 4;

__global__ void PrefixScan123(int* VectorIN, int N) {
	//int offset;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N){
		for(int level = 0; level < log2f(N); ++level){
			//printf("%f \n", floor(log2f(N)));
			int offset = pow(2, level); //2^level
			
			int index = i + offset;

            if (index < N) {
                int temp = VectorIN[index - offset];
                __syncthreads();
                VectorIN[index] += temp;
            }

            __syncthreads();
        }
	}
}

__global__ void PrefixScan(int* VectorIN, int N) {
	//int offset;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N)
		//for(int level = 0; level < ceil(log2f(N)); ++level){
		for(int offset = 1; offset < N; offset*=2){
			//printf("%f \n", ceil(log2f(N)));
			//int offset = pow(2, level); //2^level
			//int index = i + offset;
			//if (i + offset < N) {
            //if (index < N) {
			if (i >= offset) {
                int temp = VectorIN[i];
				int temp1 = VectorIN[i-offset];
                __syncthreads(); // Synchronize before updating VectorIN[i]
                VectorIN[i] = temp + temp1;
            }
            __syncthreads();

			/*if(i >= offset)
				VectorIN[i] = VectorIN[i - offset] + VectorIN[i];
			__syncthreads();
			
			if(threadIdx.x == 0 && blockIdx.x == 0){
				for (int i = 0; i < N; ++i){
					printf("%i ", VectorIN[i]);
				} printf("\n");
			}*/
		}
}

__global__ void PrefixScan1(int* VectorIN, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        for (int stride = 1; stride < N; stride *= 2) {
            int index = i + stride;

            if (index < N) {
                int temp = VectorIN[index - stride];
                __syncthreads();
                VectorIN[index] += temp;
            }

            __syncthreads();
        }
    }
}


void printArray(int* Array, int N, const char str[] = "") {
	std::cout << str;
	for (int i = 0; i < N; ++i)
		std::cout << std::setw(5) << Array[i] << ' ';
	std::cout << std::endl << std::endl;
}


#include <cstdlib>
int main(int argc, char *argv[]) {
	if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N BLOCK_SIZE" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const int BLOCK_SIZE = std::atoi(argv[2]);

	const int blockDim = BLOCK_SIZE;
	//const int N = BLOCK_SIZE * 131072;
	// ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	seed = 0;
    std::default_random_engine generator (seed);
    //std::uniform_int_distribution<int> distribution(1, 100);
	std::uniform_int_distribution<int> distribution(1, 10);

    Timer<HOST> host_TM;
    //Timer<DEVICE> dev_TM;
	Timer<HOST> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN;
	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
    SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );

	int* prefixScan = new int[N];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------
	int GridDim = ((N + blockDim - 1) / (blockDim));

	dev_TM.start();
	PrefixScan<<< GridDim, blockDim>>>(devVectorIN, N);
	dev_TM.stop();
	dev_time = dev_TM.duration();

	SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, N * sizeof(int),
                           cudaMemcpyDeviceToHost) );

	// ------------------- CUDA ENDING -----------------------------------------

	std::cout << std::fixed << std::setprecision(1)
              << "KernelTime Naive  : " << dev_time << std::endl << std::endl;

	// ------------------- VERIFY ----------------------------------------------

    host_TM.start();

	int* host_result = new int[N];
	std::partial_sum(VectorIN, VectorIN + N, host_result);

    host_TM.stop();

	
	/*printArray(VectorIN, N, "Initial");
	printArray(host_result, N, "CPU");
	printArray(prefixScan, N, "GPU");
	*/
	int flag = 0;
	for (int i = 0; i < N; ++i)
		if(host_result[i] != prefixScan[i]){
			printf("%i : %i != %i\n", i, host_result[i], prefixScan[i]);
				//printf(" ");
			flag = 1;
			//cudaDeviceReset();
			//std::exit(EXIT_FAILURE);
		}
	if(flag == 1){
		std::cerr << " Error! :  prefixScan" << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
	/*if (!std::equal(host_result, host_result + blockDim - 1, prefixScan + 1)) {
		std::cerr << " Error! :  prefixScan" << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}*/
	
    // ----------------------- SPEEDUP -----------------------------------------

    float speedup1 = host_TM.duration() / dev_time;
	std::cout << "Correct result" << std::endl
              << "(1) Speedup achieved: " << speedup1 << " x" << std::endl
              << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;
	
	delete[] host_result;
    delete[] prefixScan;
    
    SAFE_CALL( cudaFree(devVectorIN) );
    
    cudaDeviceReset();
}
