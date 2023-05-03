#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/MatrixMultiplicationToDebug.cu -I"$DIR"/include -o matrix_mul_to_debug
