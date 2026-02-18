cmake_minimum_required(VERSION 3.21)
project(NeuralSDF LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_subdirectory(external/tiny-cuda-nn)

add_executable(neural-sdf main.cu)

target_link_libraries(neural-sdf
    PRIVATE
    tiny-cuda-nn
)
