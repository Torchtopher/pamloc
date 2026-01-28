#!/bin/bash
clear
BT='Release'
if [ "$1" == "DEBUG" ]; then
    BT='Debug'
    shift
fi

BUILD_DIR="build-$BT"

if [ ! -d "$BUILD_DIR" ]; then
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BT" -S .
fi

cmake --build "$BUILD_DIR" --config "$BT"

if [ $? -ne 0 ]; then 
    echo "Build failed... exiting"
    exit 1
fi

"./$BUILD_DIR/HelloWorld" "$@"