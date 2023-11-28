#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then ## Linux
    g++ -std=c++11 -shared -fPIC -o ./src/cAdmmUpdate.so ./src/cAdmmUpdate.cpp
elif [[ "$OSTYPE" == darwin* ]]; then ## MacOS
    g++ -c -o ./src/cAdmmUpdate.o ./src/cAdmmUpdate.cpp
    g++ -shared -o ./src/cAdmmUpdate.so ./src/cAdmmUpdate.o
    rm ./src/cAdmmUpdate.o
else
    echo "ERROR: I don't know how to compile c++ and create shared library for OSTYPE=$OSTYPE; you will need to figure it out by yourself. Source files are located in ./src/cAdmmUpdate.cpp"
fi
