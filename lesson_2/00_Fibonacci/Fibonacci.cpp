#include <iostream>
#include <omp.h>

long long int fibonacci(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}

int main() {
    //  ------------------------- TEST FIBONACCI ----------------------
    omp_set_dynamic(0);
    int value = 6;

    long long int result = fibonacci(value, 1);

    std::cout << "\nresult: " << result << "\n" << std::endl;
}
