#!/usr/bin/bash 


cd build && cmake -DCMAKE_PREFIX_PATH=~/.local/src/libtorch .. && cmake --build . --config Release 
