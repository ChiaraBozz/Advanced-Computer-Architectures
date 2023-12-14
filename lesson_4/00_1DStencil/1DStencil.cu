#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int RADIUS = 7;

const int BLOCK_SIZE = 128;
const int SHMEM_SIZE = BLOCK_SIZE + 2*RADIUS;
const int N  = 100000000;


__global__ void stencilKernel(const int* d_input, int N, int* d_output) {
    __shared__ int shmem[SHMEM_SIZE];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    if(global_id < N) {
        // fill the shared memory
        int m=0;
        for(; m < SHMEM_SIZE/BLOCK_SIZE; m++){
            shmem[m*BLOCK_SIZE + tx] = d_input[m*BLOCK_SIZE + global_id];
        }
        if(tx < SHMEM_SIZE%BLOCK_SIZE){
            shmem[m*BLOCK_SIZE + tx] = d_input[m*BLOCK_SIZE + global_id];
        }
        __syncthreads(); 

        int j;
        if(global_id < N-RADIUS-1){
            for (j = threadIdx.x; j < threadIdx.x + 2*RADIUS + 1; j++){
                
                if( global_id + RADIUS < N-RADIUS){
                    d_output[global_id+RADIUS] += shmem[j];
                }
            }
        }
    }
}

__global__ void stencilKernel1(const int* d_input, int N, int* d_output) {
    __shared__ int shmem[SHMEM_SIZE];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int id_shmem = threadIdx.x;

    if(global_id < N) {
        // fill the shared memory
        // fill the block_size part
        shmem[id_shmem] = d_input[global_id];
        //printf("(GI %i, SM %i) : %i\n", global_id, id_shmem, shmem[id_shmem]);
        
        if(global_id == 4)
            printf("Finito \n\n\n");
        __syncthreads();    
        if(global_id == 4){
            for (int i = 0; i < SHMEM_SIZE; i++)
                printf("%i ", shmem[i]);
            printf("\n");
        }
        __syncthreads(); 
        
        //fill the outside parts
        if(threadIdx.x <= RADIUS){
            shmem[id_shmem + BLOCK_SIZE] = d_input[global_id + BLOCK_SIZE];
            
            printf("(GI %i, SM %i) : %i\n", global_id, id_shmem+BLOCK_SIZE, shmem[id_shmem+ BLOCK_SIZE]);
        }
        __syncthreads();
        if(global_id == 4){
            for (int i = 0; i < BLOCK_SIZE + 2*RADIUS; i++)
                printf("%i ", shmem[i]);
            printf("\n");
        }
        __syncthreads();
        //if(global_id == 0)
        //    printf("OK\n");
        
        int j;
        if(global_id < N-RADIUS-1){
        //for (j = global_id - RADIUS; j <= global_id + RADIUS; j++){
            for (j = threadIdx.x; j < threadIdx.x + 2*RADIUS + 1; j++){
                if(global_id == 4)
                    printf("J = %i \n", j);
                
                //if(threadIdx.x >= RADIUS && threadIdx.x < BLOCK_SIZE-RADIUS){
                    d_output[global_id+RADIUS] += shmem[j];
                    
                    if(global_id == 4)
                        printf("%i \n", shmem[j]);
            }
        }
    }
}

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_input      = new int[N];
    int* h_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N](); // initilization to zero
    int* h_output_zero     = new int[N](); // initilization to zero

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
        h_input[i] = distribution(generator);
    
    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = RADIUS; i < N - RADIUS; i++) {
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            h_output[i] += h_input[j];
    }

    TM_host.stop();
    TM_host.print("1DStencil host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    SAFE_CALL( cudaMalloc( &d_input, N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_output, N * sizeof(int) ));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy( d_output, h_output_zero, N * sizeof(int), cudaMemcpyHostToDevice) );

    // -------------------------------------------------------------------------
    // did you miss something?
    ///
    dim3 DimGrid(N/BLOCK_SIZE, 1, 1);
    if (N%BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    stencilKernel<<<DimGrid,DimBlock>>>(d_input, N, d_output);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("1DStencil device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_output_tmp, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << h_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_tmp;
    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_input ) )
    SAFE_CALL( cudaFree( d_output ) )
    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
