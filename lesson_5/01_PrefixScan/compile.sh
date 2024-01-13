#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScan.cu -I"$DIR"/include -o prefixscan
nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScanv2.cu -I"$DIR"/include -o workprefixscan
nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScanv2shmem.cu -I"$DIR"/include -o shmworkprefixscan
nvcc -w -std=c++11 -arch=sm_61 "$DIR"/PrefixScanv3.cu -I"$DIR"/include -o taskparprefixscan
