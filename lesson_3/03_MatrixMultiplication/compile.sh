#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_61 "$DIR"/MatrixMultiplication.cu -I"$DIR"/include -o matrix_mul
