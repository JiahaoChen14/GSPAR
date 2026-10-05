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

#include <thrust/pair.h>
#include <memory>

#include "nvblox/core/types.h"

#include <thrust/pair.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {

template <typename BlockType>
using IndexBlockPair = thrust::pair<Index3D, BlockType*>;

template <typename BlockType>
class GPUHashImpl;

template <typename BlockType>
class BlockLayer;

/// Class that manages a GPU hash
///
/// Insertion and removal operations are cached on the CPU and consolidated into
/// batches of operations that are executed when the hash is accessed. This all
/// takes place behind the scenes.
///
/// @attention: It is not supported to insert duplicated elements. Attempting to
/// do so will trigger an assertion
/// @TODO(dtingdahl) Change name of this class to GpuHashManager
template <typename BlockType>
class GPULayerView {
 public:
  using LayerType = BlockLayer<BlockType>;

  /// Number of blocks allcoated per default
  static constexpr int kDefaultCapacity = 4096;

  /// Construct a hash manager given a requested capacity
  GPULayerView(size_t max_num_blocks = kDefaultCapacity);

  GPULayerView(const GPULayerView& other) = delete;
  GPULayerView(GPULayerView&& other) = delete;
  GPULayerView& operator=(const GPULayerView& other) = delete;
  GPULayerView& operator=(GPULayerView&& other) = delete;

  ~GPULayerView();

  /// Insert several blocks
  void insertBlocksAsync(
      const std::vector<thrust::pair<Index3D, BlockType*>>& blocks_to_insert,
      const CudaStream& cuda_stream);
  /// Insert a single blocks
  void insertBlockAsync(
      const thrust::pair<Index3D, BlockType*>& block_to_insert,
      const CudaStream& cuda_stream);

  /// Remove a block
  void removeBlockAsync(const Index3D& blocks_to_remove,
                        const CudaStream& cuda_stream);
  /// Remove several blocks
  void removeBlocksAsync(const std::vector<Index3D>& blocks_to_remove,
                         const CudaStream& cuda_stream);

  /// Resize the underlying GPU hash and delete its content.
  void reset(size_t new_max_num_blocks = kDefaultCapacity,
             const CudaStream& cuda_stream = CudaStreamOwning());

  /// Get the Gpu hash.
  const GPUHashImpl<BlockType>& getHash() const { return *gpu_hash_ptr_; }

  /// Number of elements inside the hash
  size_t size() const;

  /// Max number of blocks the cache can hold
  size_t capacity() const;

  /// Max allowed load factor (size/capacity)
  float max_load_factor() const { return max_load_factor_; }

  /// When resizing the hash, it will be expanded with this amount
  float size_expansion_factor() const { return size_expansion_factor_; }

  /// Flush the internal cache. This will execute any pending insertion or
  /// removal operations.
  void flushCache(const CudaStream& cuda_stream);

  /// Return the current load factor
  float loadFactor();

  /// Check if the internal gpu hash is valid
  bool isValid(const CudaStream& cuda_stream) const;

 private:
  // Apply cached insertion operations.
  void flushInsertionCache(const CudaStream& cuda_stream);

  // Apply cached removal operations.
  void flushRemovalCache(const CudaStream& cuda_stream);

  // The load factor at which we reallocate space. Load factors of above 0.5
  // seem to cause the hash table to overfill in some cases, so please use
  // max loads lower than that.
  const float max_load_factor_ = 0.5;

  // This is the factor by which we overallocate space
  const float size_expansion_factor_ = 2.0f;

  // NOTE(alexmillane): To keep GPU code out of the header we use PIMPL to hide
  // the details of the GPU hash.
  std::shared_ptr<GPUHashImpl<BlockType>> gpu_hash_ptr_ = nullptr;

  // Used for staging when transferring the gpu hash
  device_vector<thrust::pair<Index3D, BlockType*>> blocks_to_insert_device_;
  device_vector<Index3D> blocks_to_remove_device_;

  // We cache data for insertion and removal in order to reduce the number of
  // copy-to-gpu transactions.
  mutable std::vector<thrust::pair<Index3D, BlockType*>> insertion_cache_;
  mutable std::vector<Index3D> removal_cache_;

  // Size of GPU hash, taking insertion and removal cache into account.
  size_t size_including_cache_ = 0;
};

}  // namespace nvblox

// NOTE(alexmillane):
// - I am leaving this here as a reminder to NOT include the implementation.
// - The implementation file is instead included by .cu files which declare
//   specializations.
// - The problem is that we don't want the GPULayerView implementation, which
//   contains CUDA calls and stdgpu code, bleeding into into layer.h, one of our
//   main interface headers.
// #include "nvblox/gpu_hash/internal/cuda/impl/gpu_layer_view_impl.cuh"
