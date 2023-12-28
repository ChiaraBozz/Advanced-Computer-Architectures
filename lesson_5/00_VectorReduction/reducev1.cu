/*
Vector Reduction for large arrays 
v1: without shared memory
*/

#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

//const int N  = 16777216;//27;//76744;//= 16777216;
//#define BLOCK_SIZE 256//5
#define SHMEM_SIZE 1024

__global__ void ReduceKernel(int* VectorIN, int N, int* VectorOUT) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if(globalIdx < N){
        for(int i= 1; i < blockDim.x && i < N; i*=2){
            if( threadIdx.x % (i*2) == 0)   
                if( globalIdx + i < (blockIdx.x+1) * blockDim.x && globalIdx + i < N)   {
                    //printf("\n%i: %i + %i = %i ", globalIdx, VectorIN[globalIdx], VectorIN[globalIdx + i],  VectorIN[globalIdx] + VectorIN[globalIdx + i]);
                    //VectorOUT[globalIdx] = VectorIN[globalIdx] + VectorIN[globalIdx + i];
                    VectorIN[globalIdx] = VectorIN[globalIdx] + VectorIN[globalIdx + i];
                    //shmem[ threadIdx.x ] = shmem[ threadIdx.x ] + shmem[ threadIdx.x + i];
                }
                //else if( globalIdx < (blockIdx.x+1) * blockDim.x)
                //    VectorOUT[globalIdx] = VectorIN[globalIdx];
            __syncthreads();

        }
        VectorOUT[globalIdx] = VectorIN[globalIdx];     //to avoid race conditions like:
                                                        /*  0: 0 : 450 
                                                            4: 2 : 9 
                                                            2: 1 : 458 
                                                            After0: 0 : 450 
                                                            After4: 2 : 9 
                                                            After2: 1 : 9 
                                                            final result:   450 9 9
                                                            instead of:     450 458 9    
                                                        */
        __syncthreads(); 

        if(threadIdx.x == 0){
            //printf("\n%i: %i : %i ", globalIdx, blockIdx.x, VectorOUT[globalIdx]);
            VectorIN[blockIdx.x] = VectorOUT[globalIdx];
            //printf("\nAfter%i: %i : %i ", globalIdx, blockIdx.x, VectorIN[blockIdx.x]);
        }
    }
}

//int main() {
#include <cstdlib>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N BLOCK_SIZE" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const int BLOCK_SIZE = std::atoi(argv[2]);
    // ------------------- INIT ------------------------------------------------
    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    Timer<HOST> host_TM;
    //Timer<HOST> dev_TM;
    Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i) {
		VectorIN[i] = distribution(generator);
        //printf("%i ", VectorIN[i]);
    }
	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN; 
    int* devVectorOUT;

	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
	SAFE_CALL( cudaMalloc(&devVectorOUT, N * sizeof(int)) );
	
	SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
	SAFE_CALL( cudaMemcpy(devVectorOUT, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
	
	int sum;
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();
/*
    ReduceKernel<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, N);
    
    printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE), BLOCK_SIZE, N);
    int* partialRes = new int[N];
    SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");

    ReduceKernel<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, DIV(N, BLOCK_SIZE));
    
    printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, DIV(N, BLOCK_SIZE));
    SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");
	
    ReduceKernel<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));
    //printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE,  DIV(N, BLOCK_SIZE*BLOCK_SIZE));
*/
   
    int i = BLOCK_SIZE;
    int GridDim = DIV(N, BLOCK_SIZE);
    int dimArray = N;
    do{ 
        ReduceKernel<<< GridDim , BLOCK_SIZE>>>(devVectorIN, dimArray, devVectorOUT); 
        //int* partialRes = new int[N];
        //SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
        //printf("GridDim: %i, %i\t%i\n", GridDim, i, dimArray);
        dimArray = DIV(N, i);
        i *= BLOCK_SIZE;
        GridDim = DIV(N, i);

        //for (int i = 0; i < dimArray; ++i) {printf("%i ", partialRes[i]);}printf("\n");
    }while(GridDim != 1);
    
    //printf("GridDim: %i, %i\t%i\n", GridDim, i, dimArray);
    ReduceKernel<<< GridDim , BLOCK_SIZE>>>(devVectorIN, dimArray, devVectorOUT); 
    //int* partialRes = new int[N];
    //SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
    //for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");

	dev_TM.stop();
	dev_time = dev_TM.duration();
	CHECK_CUDA_ERROR;

	SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int), cudaMemcpyDeviceToHost) );

	// ------------------- HOST ------------------------------------------------
    host_TM.start();

	int host_sum = std::accumulate(VectorIN, VectorIN + N, 0);

    host_TM.stop();

    std::cout << std::setprecision(3)
              << "KernelTime Divergent: " << dev_time << std::endl
              << "HostTime            : " << host_TM.duration() << std::endl
              << std::endl;

    // ------------------------ VERIFY -----------------------------------------

    if (host_sum != sum) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }

    //-------------------------- SPEEDUP ---------------------------------------

    float speedup = host_TM.duration() / dev_time;

    std::cout << "Correct result" << std::endl
              << "Speedup achieved: " << std::setprecision(3)
              << speedup << " x" << std::endl << std::endl;

    std::cout << host_TM.duration() << "; " << dev_TM.duration() << "; " << host_TM.duration() / dev_TM.duration() << std::endl;

    delete[] VectorIN;
    SAFE_CALL( cudaFree(devVectorIN) );
    //cudaDeviceReset();
}
