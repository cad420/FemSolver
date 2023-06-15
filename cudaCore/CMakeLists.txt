cmake_minimum_required(VERSION 3.10)

project(CudaCompute LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_NVCC_FLAGS -std=c++17)

include_directories(${PROJECT_SOURCE_DIR} )

file(GLOB CUDASOURCES ./CG.cu)

cuda_add_library(core STATIC ${CUDASOURCES})

target_link_libraries(core 
    cudart cusparse cublas)    