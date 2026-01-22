#!/bin/bash

BT='Release'
if [ "$1" == "DEBUG" ]; then
    BT='Debug'
    shift  # remove DEBUG from arguments so it doesn't get passed to program
fi

cmake -B build -DCMAKE_BUILD_TYPE=$BT -S . && cmake --build build && ./build/HelloWorld "$@"
