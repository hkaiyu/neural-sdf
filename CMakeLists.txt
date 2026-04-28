cmake_minimum_required(VERSION 3.21)
project(NeuralSDF LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

find_package(CUDAToolkit REQUIRED)

set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(external/tiny-cuda-nn)

add_executable(neural-sdf
    main.cu
    external/tiny-cuda-nn/dependencies/stbi/stbi_wrapper.cpp
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_impl_opengl3.cpp
    external/imgui/imgui_impl_rgfw.cpp
)

target_include_directories(neural-sdf
    PRIVATE
    external
    external/imgui
    external/rgfw
)

target_link_libraries(neural-sdf
    PRIVATE
    tiny-cuda-nn
    CUDA::cudart
)
