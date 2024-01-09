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

//const int BLOCK_SIZE = 4;

__global__ void PrefixScan(int* VectorIN, int N) {
	//int offset;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int step = 1;
	if(i < N)
		//for(int level = 0; level < ceil(log2f(N)); ++level){
		printf("OK");
		for(int limit = blockDim.x/2; limit > 0; limit/=2){
			if(i < limit){
				int valueRight = (i + 1) * (step * 2) - 1;
				int valueLeft = valueRight - step;
				VectorIN[valueRight] = VectorIN[valueRight] + VectorIN[valueLeft];
			}
			step *=2;			
            __syncthreads();
			/*if(threadIdx.x == 0 && blockIdx.x == 0){
				for (int i = 0; i < N; ++i){
					printf("%i ", VectorIN[i]);
				} printf("\n");
			}*/
		}
		printf("OK1");
		if(threadIdx.x == 0)
			VectorIN[blockDim.x - 1] = 0;	
        __syncthreads();
		int limit = 1;
		for(step = blockDim.x/2; step > 0; limit/=2){
			if(i < limit){
				int valueRight = (i*2 + 1) * step - 1;
				int valueLeft = valueRight - step;
				int tmp = VectorIN[valueLeft];
				VectorIN[valueLeft] = VectorIN[valueRight];
				VectorIN[valueRight] = VectorIN[valueRight] + tmp;
			}
			limit *=2;			
            __syncthreads();
		}
		printf("OK3");
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
	
	printArray(VectorIN, N, "Initial");

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
