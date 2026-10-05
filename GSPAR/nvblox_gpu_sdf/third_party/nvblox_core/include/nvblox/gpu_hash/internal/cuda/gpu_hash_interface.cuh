/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

#include <stdgpu/cstddef.h>
#include <stdgpu/unordered_map.cuh>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/internal/error_check.h"
#include "nvblox/core/types.h"
#include "nvblox/utils/logging.h"

namespace nvblox {

template <typename BlockType>
using ConstIndexBlockPair = thrust::pair<const Index3D, BlockType*>;

template <typename BlockType>
using Index3DDeviceHashMapType =
    stdgpu::unordered_map<Index3D, BlockType*, Index3DHash,
                          std::equal_to<Index3D>>;

/// Wrapper around a stdgpu hashmap
template <typename BlockType>
class GPUHashImpl {
 public:
  GPUHashImpl() = default;

  /// Construct a hashmap
  ///
  /// @param max_num_blocks  Requested capacity of the hashamp
  /// @param cuda_stream     Cuda stream for GPU work
  GPUHashImpl(int max_num_blocks, const CudaStream& cuda_stream);
  ~GPUHashImpl();

  /// Insert everything from other.
  ///
  /// @attention Container must be empty
  /// @attention Capacity must be enough
  ///
  /// @param other       Gpu hash to insert from
  /// @pram cuda_stream  Cuda stream for GPU work
  void initializeFromAsync(const GPUHashImpl<BlockType>& other,
                           const CudaStream& cuda_stream);

  ///  Copy of impl_.bucket_size() to avoid costly device-to-host mem
  /// transfer
  stdgpu::index_t max_num_blocks_;

  /// The implementation
  Index3DDeviceHashMapType<BlockType> impl_;
};

}  // namespace nvblox

#include "nvblox/gpu_hash/internal/cuda/impl/gpu_hash_interface_impl.cuh"
