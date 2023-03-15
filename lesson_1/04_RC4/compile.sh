#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -O3 -fopenmp ${DIR}/RC4.cpp -o rc4
./rc4
