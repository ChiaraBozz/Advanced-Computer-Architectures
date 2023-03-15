#!/bin/bash
DIR=`dirname $0`

#g++-9 -std=c++11 -fopenmp "$DIR"/HelloWorld.cpp -o hello_world
#g++-9 -std=c++14 -O3 -fopenmp "$DIR"/HelloWorld.cpp -o hello_world_O3

g++-9 -std=c++11 -fopenmp "$DIR"/HelloWorld.cpp -o hello_world
./hello_world

