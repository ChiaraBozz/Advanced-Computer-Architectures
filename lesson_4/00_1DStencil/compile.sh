#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_61 "$DIR"/1DStencil.cu -I"$DIR"/include -o stencil
