#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScan.cu -I"$DIR"/include -o prefixscan
nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScanv2.cu -I"$DIR"/include -o workprefixscan
#nvcc -w -std=c++11 -arch=sm_61 "$DIR"/reducev1.cu -I"$DIR"/include -o reduce_without_shmem
#nvcc -w -std=c++11 -arch=sm_61 "$DIR"/reducev2.cu -I"$DIR"/include -o reduce_shmem_less_div
#nvcc -w -std=c++11 -arch=sm_61 "$DIR"/reducev3.cu -I"$DIR"/include -o reduce_task_par
