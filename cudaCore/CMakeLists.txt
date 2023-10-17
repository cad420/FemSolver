cmake_minimum_required(VERSION 3.18)

project(CudaCompute LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 61 89)

include_directories(${PROJECT_SOURCE_DIR} )

file(GLOB CUDASOURCES ./CG.cu)

# add_library(
#     core
#     ${CUDASOURCES}
# )

add_library(core STATIC ${CUDASOURCES})

target_link_libraries(core 
   cudart cusparse cublas)   