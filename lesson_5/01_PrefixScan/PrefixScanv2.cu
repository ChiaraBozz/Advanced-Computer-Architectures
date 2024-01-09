/*
Prefix Sum for large arrays 
v2: work efficient
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

#define NUM 1024//512*512
#define BLOCK_SIZE 1024

#define NUM_BLOCKS ((NUM) + (BLOCK_SIZE) - 1) / (BLOCK_SIZE)

//__device__ int finalarray[NUM_BLOCKS-1];

__global__ void EndPrefixScan(int* VectorIN, int N, int* VectorADD) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(blockIdx.x > 0){
		//printf("%i: %i \t%i\n", tid, VectorIN[tid], VectorADD[blockIdx.x-1]);
		VectorIN[tid]+= VectorADD[blockIdx.x];
	}
}

__global__ void PrefixScan(int* VectorIN, int N, int* VectorADD) {
	//int offset;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	int DIM;
	if(N < blockDim.x){
		DIM = N;
	}else{
		DIM = blockDim.x;
	}

	/*if(threadIdx.x == 0 && blockIdx.x == 1){
		printf("INITIAL\n");
		for (int i = 0; i < N; ++i){
			printf("%i ", VectorIN[i]);
		} printf("\n");
	}
	__syncthreads();*/

  	if(tid < N){
		int step = 1;
		for(int limit = DIM/2; limit > 0; limit/=2){
			if(threadIdx.x < limit){
				int valueRight = (threadIdx.x + 1) * (step * 2) - 1+ blockDim.x*blockIdx.x;
				int valueLeft = valueRight - step;
				VectorIN[valueRight] = VectorIN[valueRight] + VectorIN[valueLeft];
			}
			step *=2;			
            __syncthreads();
			
		}
		/*if(threadIdx.x == 0 && blockIdx.x == 1){
				for (int i = 0; i < N; ++i){
					printf("%i ", VectorIN[i]);
				} printf("\n");
		}*/

		if(threadIdx.x == 0){
			VectorADD[blockIdx.x]=VectorIN[DIM*(blockIdx.x+1) - 1];
			VectorIN[DIM*(blockIdx.x+1) - 1] = 0;	

			//printf("final array: %i, %i \n", VectorADD[blockIdx.x], blockIdx.x);
		}
		__syncthreads();

		/*if(threadIdx.x == 0 && blockIdx.x == 1){
				for (int i = 0; i < N; ++i){
					printf("%i ", VectorIN[i]);
				} printf("\n");
		}*/
		
		int limit = 1;
		for(step = DIM/2; step > 0; step/=2){
				//int valueRight = (i*2 + 1) * step - 1;
				//int valueLeft = valueRight - step;
				int valueLeft = (threadIdx.x*2 + 1) * step - 1 + blockDim.x*blockIdx.x;
				int valueRight = valueLeft + step;
				int tmp = VectorIN[valueLeft];
				__syncthreads();
			if(threadIdx.x < limit){
				VectorIN[valueLeft] = VectorIN[valueRight];
				VectorIN[valueRight] = VectorIN[valueRight] + tmp;
			}
			limit *=2;			
            __syncthreads();
		}
		/*
		if(threadIdx.x == 0 && blockIdx.x == 1){
				for (int i = 0; i < N; ++i){
					printf("%i ", VectorIN[i]);
				} printf("\n");
		}*/

		/*if(blockIdx.x > 0){
			printf("%i: %i \t%i\n", tid, VectorIN[tid], finalarray[blockIdx.x-1]);
			VectorIN[tid]+= finalarray[blockIdx.x-1];
		}
		if(N/blockDim.x > 0){
			if(threadIdx.x == 0)
				finalarray[blockDim.x-1] = threadIdx.x;
		}*/
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

	int* toprint = new int[N];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------
	int GridDim = ((N + blockDim - 1) / (blockDim));

	dev_TM.start();
	PrefixScan<<< GridDim, blockDim>>>(devVectorIN, N, devVectorADD);

	cudaDeviceSynchronize();

	//printArray(toprint, (NUM_BLOCKS), "Intermediate result:\n");
	if(NUM_BLOCKS > 1){
		SAFE_CALL(cudaMemcpy(toprint, devVectorADD, (NUM_BLOCKS) * sizeof(int),
                           cudaMemcpyDeviceToHost) );
		PrefixScan<<< 1, NUM_BLOCKS>>>(devVectorADD, NUM_BLOCKS, devVectorADD1);
	}
	SAFE_CALL(cudaMemcpy(toprint, devVectorADD, (NUM_BLOCKS) * sizeof(int),
                           cudaMemcpyDeviceToHost) );
	//printArray(toprint, (NUM_BLOCKS), "Intermediate result222:\n");

	EndPrefixScan<<<GridDim, blockDim>>>(devVectorIN, N, devVectorADD);
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

	/*printArray(VectorIN, N, "Initial");
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
