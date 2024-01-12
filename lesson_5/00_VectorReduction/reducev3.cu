/*
Vector Reduction for large arrays 
v3: global with task parallelism for data transfer/kernel computation overlapping
*/

#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

//const int N  = 16777216;
//#define BLOCK_SIZE 256
#define SHMEM_SIZE 1024

// interesting config: N 536870912  BLOCK_SIZE 128  SEGSIZE 1024*1024*2

__global__ void ReduceKernel(int* VectorIN, int N) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shmem[SHMEM_SIZE];
    /*if(globalIdx == 0){
        printf("N: %i\n", N);
        for(int i = 0; i < N; i++){
            printf("%i ", VectorIN[i]);
        }printf("\n");
    }*/
    if(globalIdx < N){
        
        shmem[threadIdx.x] = VectorIN[globalIdx];
        __syncthreads();

        for(int i= 1; i < blockDim.x; i*=2){
            int idx = threadIdx.x*i*2;
            
            if(idx + i < blockDim.x && idx < blockDim.x && idx + blockIdx.x * blockDim.x + i < N)  
                shmem[ idx ] += shmem[ idx + i];
            __syncthreads();
        }
        if(threadIdx.x == 0)
            VectorIN[blockIdx.x] = shmem[threadIdx.x];
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
    int SegSize = 1024*1024*2;//*1024*64; //256;
    if(SegSize > N)
        SegSize = N;
    
    printf("%i\n", SegSize);
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
	int* h_VectorIN_tmp = new int[N];
	for (int i = 0; i < N; ++i) {
		VectorIN[i] = distribution(generator);
        h_VectorIN_tmp[i] = VectorIN[i];
        //printf("%i ", VectorIN[i]);
    }
    // ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN;
	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
	
	SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
	
	int sum;
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();

    //ReduceKernel<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, N);
    
    //printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE), BLOCK_SIZE, N);
    //int* partialRes = new int[N];
    //SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
    //for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");

    //ReduceKernel<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, DIV(N, BLOCK_SIZE));
    
    //printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, DIV(N, BLOCK_SIZE));
    //SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
    //for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");
	
    //ReduceKernel<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>(devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));
    //printf("GridDim: %i, %i\t%i\n", DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE,  DIV(N, BLOCK_SIZE*BLOCK_SIZE));

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int *d_A0; // device memory for stream 0
    int *d_A1; // device memory for stream 1

    // cudaMalloc for d_A0, d_B0, d_C0, d_A1, d_B1, d_C1 go here
    SAFE_CALL( cudaMalloc( &d_A0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_A1, SegSize * sizeof(int) ));

    int j=0;
    for (int i = 0; i < N; i += SegSize * 2) {
        cudaMemcpyAsync(d_A0, h_VectorIN_tmp + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_VectorIN_tmp + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
        //printf("indice I: %i\n", i);

        int DimGrid = DIV(SegSize, BLOCK_SIZE);
        //printf("DimGrid: %i\n", DimGrid);

        if(N-i > SegSize*2){
            //printf("i + SegSize : %i\n", i + SegSize );
            ReduceKernel<<<DimGrid, BLOCK_SIZE, 0, stream0>>>(d_A0, SegSize);
            ReduceKernel<<<DimGrid, BLOCK_SIZE, 0, stream1>>>(d_A1, SegSize);

            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(h_VectorIN_tmp + j + DimGrid, d_A1, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream1);
            //printf("SegSize / BLOCK_SIZE: %i", SegSize / BLOCK_SIZE);

            j += 2*DimGrid;
        }
        else if(N-i < SegSize){
            //printf("HAHAHHA N-i: %i\n", N-i);
            ReduceKernel<<<DimGrid, BLOCK_SIZE, 0, stream0>>>(d_A0, N-i);
            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0,  DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            j+= DimGrid;
            //ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_A1, (N-i)/2);
        }
        else{
            //printf("N-i- SegSize: %i\n", N-i- SegSize);
            //printf("i + SegSize*2: %i\n", i + SegSize*2);
            ReduceKernel<<<DimGrid, BLOCK_SIZE, 0, stream0>>>(d_A0, SegSize);
            ReduceKernel<<<DimGrid, BLOCK_SIZE, 0, stream1>>>(d_A1, N - i - SegSize);

            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(h_VectorIN_tmp + j+ DimGrid, d_A1, DimGrid * sizeof(int), cudaMemcpyDeviceToHost, stream1);

            j += 2*DimGrid;
        }
        //cudaStreamSynchronize(stream0);
        //cudaStreamSynchronize(stream1);
    }
    /* DA CANCELLARE
    int j=0;
    for (int i = 0; i < N; i += SegSize * 2) {
        cudaMemcpyAsync(d_A0, h_VectorIN_tmp + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_VectorIN_tmp + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
        //printf("indice I: %i\n", i);

        if(N-i > SegSize*2){
            //printf("i + SegSize : %i\n", i + SegSize );
            ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream0>>>(d_A0, SegSize);
            ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_A1, SegSize);

            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(h_VectorIN_tmp + j+SegSize / BLOCK_SIZE, d_A1, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream1);
            printf("SegSize / BLOCK_SIZE: %i", SegSize / BLOCK_SIZE);

            j += 2*SegSize / BLOCK_SIZE;
        }
        else if(N-i < SegSize){
            //printf("HAHAHHA N-i: %i\n", N-i);
            ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream0>>>(d_A0, N-i);
            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0,  1 * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            j+= SegSize / BLOCK_SIZE;
            //ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_A1, (N-i)/2);
        }
        else{
            //printf("N-i- SegSize: %i\n", N-i- SegSize);
            //printf("i + SegSize*2: %i\n", i + SegSize*2);
            ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream0>>>(d_A0, SegSize);
            ReduceKernel<<<SegSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_A1, N - i - SegSize);

            cudaMemcpyAsync(h_VectorIN_tmp + j, d_A0, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(h_VectorIN_tmp + j+SegSize / BLOCK_SIZE, d_A1, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream1);

            j += 2*SegSize / BLOCK_SIZE;
        }
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
    }
    */
   sum = std::accumulate(h_VectorIN_tmp, h_VectorIN_tmp + j, 0);
   /*FATTO 2a PARTE IN GPU
    SAFE_CALL( cudaMemcpy(devVectorIN, h_VectorIN_tmp, N * sizeof(int), cudaMemcpyHostToDevice) ); //to refine
    
    j = DIV(N, BLOCK_SIZE);
    //j = DIV(N, BLOCK_SIZE);

    //for(int i = 0; i < j; i ++ )        printf("%i ", h_VectorIN_tmp[i]);    printf("\n");
    printf("j: %i\n", j);
    //ReduceKernel<<<1, j>>>(devVectorIN, j);

    int i = BLOCK_SIZE;
    int GridDim = DIV(j, BLOCK_SIZE);
    int dimArray = j;
    do{  
        ReduceKernel<<< GridDim , BLOCK_SIZE>>>(devVectorIN, dimArray); 
        //printf("GridDim: %i, %i\t%i\n", GridDim, i, dimArray);
        
        dimArray = DIV(j, i);
        i *= BLOCK_SIZE;
        GridDim = DIV(j, i);
        //int* partialRes = new int[N];
        //SAFE_CALL( cudaMemcpy(partialRes, devVectorIN,  N * sizeof(int), cudaMemcpyDeviceToHost) );
        //for (int i = 0; i < N; ++i) {printf("%i ", partialRes[i]);}printf("\n");
        //for (int i = dimArray-1; i < dimArray+5; ++i) {printf("%i ", partialRes[i]);}printf("\n");
    }while(GridDim != 1);
    //printf("GridDim: %i, %i\t%i\n", GridDim, i, dimArray);
    ReduceKernel<<< GridDim , BLOCK_SIZE>>>(devVectorIN, dimArray); 
    */
	dev_TM.stop();
	dev_time = dev_TM.duration();
	CHECK_CUDA_ERROR;

	//SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int), cudaMemcpyDeviceToHost) );

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
    // -------------------------------------------------------------------------
    // DESTROY STREAMS
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    //cudaDeviceReset();
}
