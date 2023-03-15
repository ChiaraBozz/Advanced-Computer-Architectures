#!/bin/bash
#range of IPs Gamma Desktops: 10.220.32.51 - 10.220.32.72
#e.g.: ssh -XC username@10.220.32.51

DIR=`dirname $0`

export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin:$PATH

nvcc -w -std=c++11 "$DIR"/VectorAdd.cu -I"$DIR"/include -o vector_add
