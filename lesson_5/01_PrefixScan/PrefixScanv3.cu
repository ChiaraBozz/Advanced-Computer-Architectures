/*
Prefix Sum for large arrays 
v3: global with task parallelism for data transfer/kernel computation overlapping
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

#define NUM 1024*1024*256
#define BLOCK_SIZE 128//1024
#define SEGSIZE 1024*1024*2 //512*512

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

int main(int argc, char *argv[]) {

    const int N = NUM;

	const int blockDim = BLOCK_SIZE;
	int SegSize = SEGSIZE; //512;
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
	int* VectorIN_tmp = new int[N];
	for (int i = 0; i < N; ++i){
		VectorIN[i] = distribution(generator);
		VectorIN_tmp[i] = VectorIN[i];
	}
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

	//printf("GridDim: %i\n", GridDim);
	//printf("blockDim: %i\n", blockDim);

	dev_TM.start();
	
	cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int *d_A0; // device memory for stream 0
    int *d_A1; // device memory for stream 1

	int *d_Add0; // device memory for stream 0
    int *d_Add1; // device memory for stream 1

    // cudaMalloc for d_A0, d_B0, d_C0, d_A1, d_B1, d_C1 go here
    SAFE_CALL( cudaMalloc( &d_A0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_A1, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_Add0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_Add1, SegSize * sizeof(int) ));

    int j=0;
	int DimGrid = DIV(SegSize, BLOCK_SIZE);
	printf("SegSize: %i\n", SegSize);
	printf("BLOCK_SIZE: %i\n", BLOCK_SIZE);
	printf("DimGrid: %i\n", DimGrid);

    for (int i = 0; i < N; i += SegSize * 2) {
        cudaMemcpyAsync(d_A0, VectorIN + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, VectorIN + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
        
		//printArray(VectorIN + i, SegSize, "d_A0");
		//printArray(VectorIN + i + SegSize, SegSize, "d_A1");

        PrefixScan<<<DimGrid, BLOCK_SIZE, 0, stream0>>>(d_A0, SegSize, d_Add0);
        PrefixScan<<<DimGrid, BLOCK_SIZE, 0, stream1>>>(d_A1, SegSize, d_Add1);

        cudaMemcpyAsync(VectorIN_tmp + i, d_A0, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(VectorIN_tmp + i + SegSize, d_A1, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        
		//printArray(VectorIN_tmp + i, SegSize, "tmp0");
		//printArray(VectorIN_tmp + i + SegSize, SegSize, "tmp1");

		cudaMemcpyAsync(VectorADD + j, d_Add0, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(VectorADD + j + DimGrid, d_Add1, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream1);
            
        j += 2*DimGrid;
         
        //cudaStreamSynchronize(stream0);
        //cudaStreamSynchronize(stream1);
    }
	/*printf("j: %i\n", j);
	printArray(VectorADD, j, "ADD");
	printArray(VectorIN, N, "INI");
	printArray(VectorIN_tmp, N, "TMP");*/
	
	/**/
	std::partial_sum(VectorADD, VectorADD + j, VectorADD1);
	//printArray(VectorADD1, j, "2AD");
    SAFE_CALL( cudaMemcpy(devVectorADD1, VectorADD1, j * sizeof(int), cudaMemcpyHostToDevice) );
	SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN_tmp, N * sizeof(int), cudaMemcpyHostToDevice) );
		
	EndPrefixScan<<<GridDim, blockDim>>>(devVectorIN, N, devVectorADD1);
	
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

	//printArray(host_result, N, "CPU");
	//printArray(prefixScan, N, "GPU");
	
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
