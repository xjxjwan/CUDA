cmake_minimum_required(VERSION 3.29)
project(GPU CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(GPU main.cu)

set_target_properties(GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
