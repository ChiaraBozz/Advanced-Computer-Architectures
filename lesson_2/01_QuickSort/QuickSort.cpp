#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>

// ---------- DO NOT MODIFY ----------------------------------------------------
int partition(int array[], const int start, const int end) {
    int p = start;
    int pivot_element = array[end];
    for(int i = start; i < end; i++) {
        if(array[i] < pivot_element) {
            std::swap(array[i], array[p]);
            ++p;
        }
    }
    std::swap(array[p], array[end]);
    return p;
}
//------------------------------------------------------------------------------

void quick_sort(int array[], const int start, const int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
        quick_sort(array, start, pivot - 1);
        quick_sort(array, pivot + 1, end);
    }
}

template<typename T>
void print_array(T* array, int size, const char* str) {
    std::cout << str << "\n";
    for (int i = 0; i < size; ++i)
        std::cout << array[i] << ' ';
    std::cout << "\n" << std::endl;
}


int main() {
    const int N = 100;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    int* input = new int[N];
    for (int i = 0; i < N; ++i)
        input[i] = distribution(generator);

    print_array(input, N, "\nInput:");

    quick_sort(input, 0, N - 1);

    print_array(input, N, "Sorted:");
}
