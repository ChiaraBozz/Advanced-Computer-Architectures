#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_61 "$DIR"/reduce.cu -I"$DIR"/include -o reduce
nvcc -w -std=c++11 -arch=sm_61 "$DIR"/reducev1.cu -I"$DIR"/include -o reduce_without_shmem
