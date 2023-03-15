
#define __CUDA_ERROR(msg)                                      \
    {                                                          \
        cudaDeviceSynchronize();                               \
        __getLastCudaError(msg, __FILE__, __LINE__);           \
    }

#define __SAFE_CALL(function)                                  \
    {                                                          \
        __safe_call(function, __FILE__, __LINE__);             \
    }

inline
void __getLastCudaError(const char *errorMessage,
                        const char *file,
                        int line);

inline
void __safe_call(cudaError_t err, const char *file, int line);

#include "cuda_err.i.cuh"
