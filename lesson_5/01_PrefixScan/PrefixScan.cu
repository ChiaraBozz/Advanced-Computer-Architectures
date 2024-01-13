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

// It works properly only if BLOCK_SIZE < N
__global__ void PrefixScan_1(int* VectorIN, int N) {
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
		}
}

__global__ void PrefixScan(int* VectorIN, int N, int* VectorAdd) {
	//int offset;
	__shared__ int shMem[1024];
 	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int i = threadIdx.x;

	shMem[threadIdx.x] = VectorIN[tid];
	__syncthreads();

	if(tid < N){
		for(int offset = 1; offset < blockDim.x; offset*=2){
			if (i >= offset) {
                int temp = shMem[i];
				int temp1 = shMem[i-offset];
                __syncthreads(); // Synchronize before updating VectorIN[i]
                shMem[i] = temp + temp1;
            }
            __syncthreads();
		}
		VectorIN[tid] = shMem[threadIdx.x];
		if(i == blockDim.x -1)
			VectorAdd[blockIdx.x] = shMem[i];
	}
}

__global__ void PrefixScangg(int* VectorIN, int N, int* VectorAdd) {
    __shared__ int shMem[1024 + 1];  // Add padding to avoid bank conflicts
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;

    shMem[i] = (tid < N) ? VectorIN[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int temp = 0;
        if (i >= offset) {
            temp = shMem[i - offset];
        }
        __syncthreads();

        if (i >= offset) {
            shMem[i] += temp;
        }
        __syncthreads();
    }

    if (tid < N) {
        VectorIN[tid] = shMem[i];
        if (i == blockDim.x - 1) {
            VectorAdd[blockIdx.x] = shMem[blockDim.x];  // Store the block sum
        }
    }
}



__global__ void EndPrefixScan(int* VectorIN, int N, int* VectorADD) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//int i = (tid) * (blockIdx.x);
	if(blockIdx.x > 0){
		int toSum = VectorADD[blockIdx.x-1];
	
		if(tid<N)
			//printf("%i : %i, ", tid, blockIdx.x);
			//if(blockIdx.x > 0){
				//printf("\n%i : %i + %i\n", tid, VectorIN[tid], toSum);
				VectorIN[tid]+= toSum;
			//}
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
	const int GridDim = ((N + blockDim - 1) / (blockDim));
	//const int N = BLOCK_SIZE * 131072;
	// ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
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
	int* VectorADD = new int[GridDim];
	int* VectorADD1 = new int[GridDim];
	int* devVectorIN;
	int* devVectorADD;
	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&devVectorADD, GridDim * sizeof(int)) );
    SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );

	int* prefixScan = new int[N];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------
	

	dev_TM.start();
	PrefixScan<<< GridDim, blockDim>>>(devVectorIN, N, devVectorADD);
	
	//cudaDeviceSynchronize();
	//printArray(VectorIN, N, "INI");
	//SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, N * sizeof(int),
    //                       cudaMemcpyDeviceToHost) );
	//printArray(prefixScan, N, "2GP");

	printf("GridDim: %i\n", GridDim);
	if(GridDim > 1){
		SAFE_CALL(cudaMemcpy(VectorADD, devVectorADD, GridDim * sizeof(int),cudaMemcpyDeviceToHost) );
		
		//printArray(VectorADD, GridDim, "VAD");
		std::partial_sum(VectorADD, VectorADD + GridDim, VectorADD1);
		
		//printArray(VectorADD1, GridDim, "VA1");
		SAFE_CALL( cudaMemcpy(devVectorADD, VectorADD1, GridDim * sizeof(int), cudaMemcpyHostToDevice) );
		EndPrefixScan<<< GridDim, blockDim>>>(devVectorIN, N, devVectorADD);
	}
	/*
	if(GridDim > 1){
		SAFE_CALL(cudaMemcpy(VectorADD, devVectorADD, NUM_BLOCKS * sizeof(int),cudaMemcpyDeviceToHost) );
		std::partial_sum(VectorADD, VectorADD + NUM_BLOCKS, VectorADD1);
		//int NewGridDim = ((NUM_BLOCKS + blockDim - 1) / (blockDim));
		//PrefixScan<<< NewGridDim, blockDim>>>(devVectorADD, NUM_BLOCKS, devVectorADD1); // -> an illegal memory access was encountered(700)
		SAFE_CALL( cudaMemcpy(devVectorADD1, VectorADD1, NUM_BLOCKS * sizeof(int), cudaMemcpyHostToDevice) );
		EndPrefixScan<<<GridDim, blockDim>>>(devVectorIN, N, devVectorADD1);
	}*/

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

	
	//printArray(VectorIN, N, "INI");
	//printArray(host_result, N, "CPU");
	//printArray(prefixScan, N, "GPU");
	
	int flag = 0;
	for (int i = 0; i < N; ++i)
		if(host_result[i] != prefixScan[i]){
			printf("%i : %i != %i\n", i, host_result[i], prefixScan[i]);
				//printf(" ");
			flag = 1;
			//cudaDeviceReset();
			std::exit(EXIT_FAILURE);
		}
	if(flag == 1){
		std::cerr << " Error! :  prefixScan" << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}	
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
