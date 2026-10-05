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

#include "nvblox/core/internal/error_check.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/utils/timing.h"

namespace nvblox {

template <typename BlockType>
GPULayerView<BlockType>::GPULayerView(size_t max_num_blocks) {
  reset(max_num_blocks, CudaStreamOwning());
}

template <typename BlockType>
GPULayerView<BlockType>::~GPULayerView() {
  // The GPUHashImpl takes care of cleaning up GPU memory.
}

template <typename BlockType>
void GPULayerView<BlockType>::insertBlocksAsync(
    const std::vector<thrust::pair<Index3D, BlockType*>>& blocks_to_insert,
    const CudaStream& cuda_stream) {
  // By design, we do not allow the insertion and removal caches to both contain
  // elements at the same time. This is to avoid conflicts, e.g. when the same
  // index appear in both insertion and removal cache. We therefore flush the
  // insertion cache before inserting.
  flushRemovalCache(cuda_stream);

  insertion_cache_.reserve(insertion_cache_.size() + blocks_to_insert.size());
  for (const auto& item : blocks_to_insert) {
    insertion_cache_.push_back(item);
  }
  size_including_cache_ += blocks_to_insert.size();
}

template <typename BlockType>
void GPULayerView<BlockType>::insertBlockAsync(
    const thrust::pair<Index3D, BlockType*>& block_to_insert,
    const CudaStream& cuda_stream) {
  flushRemovalCache(cuda_stream);
  insertion_cache_.push_back(block_to_insert);
  ++size_including_cache_;
}
template <typename BlockType>
void GPULayerView<BlockType>::flushCache(const CudaStream& cuda_stream) {
  CHECK(insertion_cache_.empty() || removal_cache_.empty());

  flushInsertionCache(cuda_stream);
  flushRemovalCache(cuda_stream);

  // Note(dtingdahl): The currently used gpu-hash backend does not support
  // asynchronous insertion. We stil add an explicit sync here to avoid
  // surprises if this behavior changes in the future.
  cuda_stream.synchronize();
}

template <typename BlockType>
float GPULayerView<BlockType>::loadFactor() {
  return static_cast<float>(size_including_cache_) /
         gpu_hash_ptr_->max_num_blocks_;
}

template <typename BlockType>
void GPULayerView<BlockType>::flushInsertionCache(
    const CudaStream& cuda_stream) {
  if (insertion_cache_.empty()) {
    return;
  }

  // Sanity check
  CHECK_EQ(size_including_cache_,
           gpu_hash_ptr_->impl_.size() + insertion_cache_.size());

  timing::Timer timer("gpu_hash/flush_insertion_cache");
  CHECK_NOTNULL(gpu_hash_ptr_);

  if (loadFactor() > max_load_factor_) {
    // Create a new hash with the required capacity
    const size_t new_max_num_blocks = static_cast<size_t>(
        std::ceil(size_expansion_factor_ *
                  std::max<size_t>(size_including_cache_,
                                   gpu_hash_ptr_->max_num_blocks_)));

    LOG(INFO) << "Resizing GPU hash capacity from "
              << gpu_hash_ptr_->max_num_blocks_ << " to " << new_max_num_blocks
              << " in order to accomodate space for " << insertion_cache_.size()
              << " new elements.";

    auto new_gpu_hash = std::make_shared<GPUHashImpl<BlockType>>(
        new_max_num_blocks, cuda_stream);

    CHECK_GE(new_gpu_hash->max_num_blocks_, gpu_hash_ptr_->max_num_blocks_);

    // Copy everything from the old hash into the new one and swap'em
    new_gpu_hash->initializeFromAsync(*gpu_hash_ptr_, cuda_stream);
    std::swap(gpu_hash_ptr_, new_gpu_hash);
  }

  // Copy blocks from cache to device and insert them
  blocks_to_insert_device_.copyFromAsync(insertion_cache_, cuda_stream);
  CHECK(static_cast<size_t>(gpu_hash_ptr_->impl_.max_size()) >=
        gpu_hash_ptr_->impl_.size() + blocks_to_insert_device_.size());
  gpu_hash_ptr_->impl_.insert(
      thrust::device.on(cuda_stream),
      stdgpu::make_device(blocks_to_insert_device_.data()),
      stdgpu::make_device(blocks_to_insert_device_.data() +
                          blocks_to_insert_device_.size()));
  insertion_cache_.clear();
  cuda_stream.synchronize();

  // This check fails if duplicated entries are inserted into the hash. The hash
  // would then contain less elements than the size variable.
  CHECK_EQ(size_including_cache_,
           static_cast<size_t>(gpu_hash_ptr_->impl_.size()));

}  // namespace nvblox

template <typename BlockType>
void GPULayerView<BlockType>::removeBlocksAsync(
    const std::vector<Index3D>& blocks_to_remove,
    const CudaStream& cuda_stream) {
  flushInsertionCache(cuda_stream);
  removal_cache_.reserve(removal_cache_.size() + blocks_to_remove.size());
  for (const auto& item : blocks_to_remove) {
    removal_cache_.push_back(item);
  }
  size_including_cache_ -= blocks_to_remove.size();
}
template <typename BlockType>
void GPULayerView<BlockType>::removeBlockAsync(const Index3D& block_to_remove,
                                               const CudaStream& cuda_stream) {
  flushInsertionCache(cuda_stream);
  removal_cache_.push_back(block_to_remove);
  --size_including_cache_;
}

template <typename BlockType>
void GPULayerView<BlockType>::flushRemovalCache(const CudaStream& cuda_stream) {
  if (removal_cache_.empty()) {
    return;
  }

  timing::Timer timer("gpu_hash/flush_removal_cache");

  CHECK_EQ(size_including_cache_,
           gpu_hash_ptr_->impl_.size() - removal_cache_.size());
  CHECK_NOTNULL(gpu_hash_ptr_);

  // Copy removal cache to GPU and erase them
  blocks_to_remove_device_.copyFromAsync(removal_cache_, cuda_stream);
  gpu_hash_ptr_->impl_.erase(
      thrust::device.on(cuda_stream),
      stdgpu::make_device(blocks_to_remove_device_.data()),
      stdgpu::make_device(blocks_to_remove_device_.data() +
                          blocks_to_remove_device_.size()));
  removal_cache_.clear();
  cuda_stream.synchronize();
}

template <typename BlockType>
void GPULayerView<BlockType>::reset(size_t new_max_num_blocks,
                                    const CudaStream& cuda_stream) {
  timing::Timer timer("gpu_hash/transfer/reset");

  gpu_hash_ptr_ =
      std::make_shared<GPUHashImpl<BlockType>>(new_max_num_blocks, cuda_stream);
  gpu_hash_ptr_->impl_.clear();
  insertion_cache_.clear();
  removal_cache_.clear();
  size_including_cache_ = 0;
}

template <typename BlockType>
size_t GPULayerView<BlockType>::size() const {
  return size_including_cache_;
}

template <typename BlockType>
size_t GPULayerView<BlockType>::capacity() const {
  return gpu_hash_ptr_->max_num_blocks_;
}

template <typename BlockType>
bool GPULayerView<BlockType>::isValid(const CudaStream& cuda_stream) const {
  return gpu_hash_ptr_->impl_.valid(thrust::device.on(cuda_stream));
}

}  // namespace nvblox