/*
Copyright 2024 NVIDIA CORPORATION

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

#include <memory>
#include "nvblox/core/types.h"

namespace nvblox {

/// Iterator over voxels in a Voxelblock.
template <typename VoxelType, int VoxelsPerSide, bool IsConst>
class VoxelIterator {
 public:
  /// Some traits needed for this to become an iterator
  using iterator_category = std::forward_iterator_tag;
  using value_type = VoxelType;
  using difference_type = std::ptrdiff_t;
  using pointer =
      typename std::conditional<IsConst, const VoxelType*, VoxelType*>::type;
  using reference =
      typename std::conditional<IsConst, const VoxelType&, VoxelType>::type;

  /// Construct from a pointer
  VoxelIterator(pointer ptr = nullptr) : ptr_(ptr), idx_(0) {}
  ~VoxelIterator() = default;

  /// Accessors
  VoxelType& operator*() { return *ptr_; }
  const VoxelType& operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }

  /// Prefix increment
  VoxelIterator operator++() {
    ++ptr_;
    ++idx_;
    return *this;
  }

  // Postfix increment
  VoxelIterator operator++(int) {
    VoxelIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  /// Comparison operators
  friend bool operator==(const VoxelIterator& a, const VoxelIterator& b) {
    return a.ptr_ == b.ptr_;
  };
  friend bool operator!=(const VoxelIterator& a, const VoxelIterator& b) {
    return a.ptr_ != b.ptr_;
  };

  /// Get the current voxel index
  Index3D index() const {
    return {idx_ / (VoxelsPerSide * VoxelsPerSide),
            (idx_ / VoxelsPerSide) % VoxelsPerSide, idx_ % VoxelsPerSide};
  }

 private:
  pointer ptr_;
  int idx_;
};

}  // namespace nvblox
