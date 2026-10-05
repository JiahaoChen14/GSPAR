cmake_minimum_required(VERSION 3.22)

project(nvblox_core_minimal LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED CONFIG)

add_library(nvblox_gpu_hash_minimal STATIC
  src/core/error_check.cu
  src/gpu_hash/esdf_layer_specialization.cu
  src/gpu_hash/freespace_layer_specialization.cu
  src/gpu_hash/gpu_layer_view.cu
  src/gpu_hash/gpu_set.cu
  src/gpu_hash/occupancy_layer_specialization.cu
  src/gpu_hash/tsdf_layer_specialization.cu
  src/utils/nvtx_ranges.cpp
  src/utils/timing.cpp
)

add_library(nvblox_minimal SHARED
  src/core/cuda_stream.cpp
  src/core/error_check.cu
  src/core/parameter_tree.cpp
  src/geometry/bounding_boxes.cpp
  src/geometry/bounding_spheres.cpp
  src/geometry/transforms.cpp
  src/geometry/workspace_bounds.cpp
  src/integrators/esdf_integrator.cu
  src/integrators/projective_tsdf_integrator.cu
  src/integrators/view_calculator.cu
  src/map/blox.cu
  src/map/layer.cu
  src/sensors/camera.cpp
  src/sensors/image.cu
  src/utils/nvtx_ranges.cpp
  src/utils/timing.cpp
)

foreach(target nvblox_gpu_hash_minimal nvblox_minimal)
  target_compile_features(${target} PUBLIC cxx_std_17 cuda_std_17)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )
  target_compile_definitions(${target}
    PUBLIC
      CUB_WRAPPED_NAMESPACE=nvblox
      C10_USE_GLOG=1
  )
  target_compile_options(${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--display_error_number>
      $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20054>
      $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20091>
      $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=737>
  )
  set_target_properties(${target}
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION OFF
  )
endforeach()

target_link_libraries(nvblox_gpu_hash_minimal
  PUBLIC
    Eigen3::Eigen
    stdgpu::stdgpu
    CUDA::cudart
  PRIVATE
    glog::glog
)

target_link_libraries(nvblox_minimal
  PUBLIC
    Eigen3::Eigen
    CUDA::cudart
    nvblox_gpu_hash_minimal
  PRIVATE
    glog::glog
)

if(TARGET CUDA::nvToolsExt)
  target_link_libraries(nvblox_gpu_hash_minimal PUBLIC CUDA::nvToolsExt)
  target_link_libraries(nvblox_minimal PUBLIC CUDA::nvToolsExt)
endif()
