# WIP Neural SDF Training + Rendering

A small application written in CUDA using tiny-cuda-nn and OpenGL for visualizing real-time training
and learning of hash-encoded neural SDFs. 

# Build Instructions

An Nvidia GPU that supports CUDA 11.5+ is required. So far, I have only tested on Windows, and 
cannot guarantee it works on Linux. The only external dependency needed is CUDA Toolkit; all other dependencies are vendored
in this repo.

This project uses CMake. To build, go to project root and run:

```
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cd build
cmake --build .
```
