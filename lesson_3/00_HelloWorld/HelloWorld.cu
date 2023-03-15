#include <iostream>

__global__
void helloWorldKernel() {
    printf("Hello from Device, thread: %d\n", threadIdx.x);
}



int main() {
    std::cout << "(1) Hello from Host" << std::endl;

    helloWorldKernel<<< 2, 8 >>>();  // asynchronous call

    cudaDeviceSynchronize();
    std::cout << "(2) Hello from Host" << std::endl;

    
    cudaDeviceReset();
}
