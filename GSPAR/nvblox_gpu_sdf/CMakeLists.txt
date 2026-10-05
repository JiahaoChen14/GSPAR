cmake_minimum_required(VERSION 3.22)

project(nvblox_gpu_sdf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

if(DEFINED ENV{CONDA_PREFIX} AND IS_DIRECTORY "$ENV{CONDA_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(glog REQUIRED CONFIG)

set(STDGPU_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(STDGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(STDGPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(STDGPU_ENABLE_CONTRACT_CHECKS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/stdgpu)
set_target_properties(stdgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(third_party/nvblox_core)

add_library(nvblox_gpu_sdf_lib
  src/dataset_io.cpp
  src/gpu_sdf_mapper.cpp
)

target_include_directories(nvblox_gpu_sdf_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nvblox_gpu_sdf_lib
  PUBLIC
    nvblox_minimal
    glog::glog
)

add_executable(nvblox_gpu_sdf_offline
  src/main.cpp
)

target_link_libraries(nvblox_gpu_sdf_offline PRIVATE nvblox_gpu_sdf_lib)
