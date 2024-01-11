/*
Prefix Sum for large arrays 
v2: work efficient with shared memory
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

#define NUM 1024*1024*512
#define BLOCK_SIZE 1024

#define NUM_BLOCKS ((NUM) + (BLOCK_SIZE) - 1) / (BLOCK_SIZE)

__global__ void EndPrefixScan(int* VectorIN, int N, int* VectorADD) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N)
		if(blockIdx.x > 0 && blockIdx.x < NUM_BLOCKS){
			//printf("%i: %i \t%i\n", tid, VectorIN[tid], VectorADD[blockIdx.x-1]);
			VectorIN[tid]+= VectorADD[blockIdx.x-1];
		}
}

__global__ void PrefixScan(int* VectorIN, int N, int* VectorADD) {
 	__shared__ int shMem[1024];
 	int tid = blockIdx.x * blockDim.x + threadIdx.x;

 	shMem[threadIdx.x] = VectorIN[tid];
	__syncthreads();
	if(tid < N){
		int step = 1;
		for (int limit = blockDim.x / 2; limit > 0; limit /= 2) {
			if (threadIdx.x < limit) {
				int valueRight = (threadIdx.x + 1) * (step * 2) - 1;
				int valueLeft = valueRight - step;
				shMem[valueRight] += shMem[valueLeft];
			}
			step *= 2;
			__syncthreads();
		}
		if (threadIdx.x == 0){
			VectorADD[blockIdx.x]=shMem[blockDim.x - 1];
			shMem[blockDim.x - 1] = 0;
		}
		__syncthreads();

		int limit = 1;
		for(step = blockDim.x/2; step > 0; step/=2){
			if (threadIdx.x < limit) {
				int valueRight = (threadIdx.x + 1) * (step * 2) - 1;
				int valueLeft = valueRight - step;
				int tmp = shMem[valueLeft];
				shMem[valueLeft] = shMem[valueRight];
				shMem[valueRight] += tmp;
			}
			limit *=2;	
			__syncthreads();
		}
		VectorIN[tid] = shMem[threadIdx.x];
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

    const int N = NUM;

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
	int* devVectorADD;
	int* devVectorADD1;
	printf("N: %i\n", N);
	printf("NUM_BLOCKS: %i\n", NUM_BLOCKS);

	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
	SAFE_CALL( cudaMalloc(&devVectorADD, (NUM_BLOCKS) * sizeof(int)) );
	SAFE_CALL( cudaMalloc(&devVectorADD1, (NUM_BLOCKS) * sizeof(int)) );
	
	//printArray(VectorIN, N, "Initial");
    SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
	int* prefixScan = new int[N];

	int* VectorADD = new int[NUM_BLOCKS];
	int* VectorADD1 = new int[NUM_BLOCKS];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------
	int GridDim = ((N + blockDim - 1) / (blockDim));

	printf("GridDim: %i\n", GridDim);
	printf("blockDim: %i\n", blockDim);

	dev_TM.start();
	PrefixScan<<< GridDim, blockDim>>>(devVectorIN, N, devVectorADD);

	//cudaDeviceSynchronize();
	//printArray(toprint, (NUM_BLOCKS), "Intermediate result:\n");
	if(NUM_BLOCKS > 1){
		//SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost) );
		SAFE_CALL(cudaMemcpy(VectorADD, devVectorADD, NUM_BLOCKS * sizeof(int),cudaMemcpyDeviceToHost) );
		std::partial_sum(VectorADD, VectorADD + NUM_BLOCKS, VectorADD1);
		//int NewGridDim = ((NUM_BLOCKS + blockDim - 1) / (blockDim));
		//PrefixScan<<< NewGridDim, blockDim>>>(devVectorADD, NUM_BLOCKS, devVectorADD1); // -> an illegal memory access was encountered(700)
		SAFE_CALL( cudaMemcpy(devVectorADD1, VectorADD1, NUM_BLOCKS * sizeof(int), cudaMemcpyHostToDevice) );
		EndPrefixScan<<<GridDim, blockDim>>>(devVectorIN, N, devVectorADD1);
	}
	//SAFE_CALL(cudaMemcpy(toprint, devVectorADD, (NUM_BLOCKS) * sizeof(int), cudaMemcpyDeviceToHost) );
	//printArray(toprint, (NUM_BLOCKS), "Intermediate result222:\n");
	//SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, N * sizeof(int), cudaMemcpyDeviceToHost) );

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

	// Exclusive prefix sum
	for(int i = N; i>0; i--){
		host_result[i] = host_result[i-1]; 
	}
	host_result[0] = 0;

    host_TM.stop();

	/*printArray(VectorIN, N, "INI");
	printArray(host_result, N, "CPU");
	printArray(prefixScan, N, "GPU");
	*/
	int flag = 0;
	for (int i = 0; i < N; ++i)
		if(host_result[i] != prefixScan[i]){
			//printf("%i : %i != %i\n", i, host_result[i], prefixScan[i]);
			
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
