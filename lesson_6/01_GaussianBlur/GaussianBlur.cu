#include <chrono>
#include <iomanip>
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    
    // -------------------------------------------------------------------------
    // READ INPUT IMAGE


    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION


    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    /// GaussianBlurHost();

    TM_host.stop();
    TM_host.print("GaussianBlur host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    /// GaussianBlurDevice<<<  >>>();

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("GaussianBlur device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST


    // -------------------------------------------------------------------------
    // RESULT CHECK
    if (true /* Correct check here */) {
        std::cerr << "wrong result!" << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION


    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION


    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
