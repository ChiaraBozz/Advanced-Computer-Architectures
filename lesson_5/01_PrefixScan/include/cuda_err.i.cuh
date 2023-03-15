
inline
void __getLastCudaError(const char *errorMessage,
                        const char *file,
                        int line) {

	cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        std::cerr << std::endl << " CUDA error   "
                  << file << "(" << line << ")"
                  << " : " << errorMessage
                  << " -> " << cudaGetErrorString(err) << "("
                  << static_cast<int>(err)  << ") "<< std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
}

inline
void __safe_call(cudaError_t err, const char *file, int line) {
    if (cudaSuccess != err) {
        std::cerr << std::endl << " CUDA error   "
                  << file  << "(" << line << ")"
                  << " -> " << cudaGetErrorString(err) << "("
                  << static_cast<int>(err) << ") "<< std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
}
