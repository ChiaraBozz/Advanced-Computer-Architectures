set(PROJECT "MatrixMultiplication")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# ------------------------------------------------------------------------------

if(CUDA_VERSION_MAJOR LESS 7)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v7"
                        " Founded: Cuda Toolkit v${CUDA_VERSION_MAJOR}")
endif()

# ------------------------------------------------------------------------------
set (CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-arch=sm_30" "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(PrefixScan
                   ${PROJECT_SOURCE_DIR}/PrefixScan.cu
                   ${PROJECT_SOURCE_DIR}/src/Timer.cpp
                   ${PROJECT_SOURCE_DIR}/src/Timer.cu)
