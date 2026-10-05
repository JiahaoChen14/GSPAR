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

#include "nvblox/utils/logging.h"

#include "nvblox/map/blox.h"

namespace nvblox {

constexpr float voxelSizeToBlockSize(const float voxel_size_m) {
  return voxel_size_m * VoxelBlock<bool>::kVoxelsPerSide;
}

constexpr float blockSizeToVoxelSize(const float block_size_m) {
  constexpr float kVoxelsPerSideInv = 1.0f / VoxelBlock<bool>::kVoxelsPerSide;
  return block_size_m * kVoxelsPerSideInv;
}

Index3D getBlockIndexFromPositionInLayer(const float block_size_m,
                                         const Vector3f& position) {
  Eigen::Vector3i index = (position / block_size_m).array().floor().cast<int>();
  return Index3D(index.x(), index.y(), index.z());
}

void getBlockAndVoxelIndexFromPositionInLayer(const float block_size_m,
                                              const Vector3f& position,
                                              Index3D* block_idx,
                                              Index3D* voxel_idx) {
  constexpr int kVoxelsPerSideMinusOne = VoxelBlock<bool>::kVoxelsPerSide - 1;
  const float voxel_size_m_inv = 1.0 / blockSizeToVoxelSize(block_size_m);
  *block_idx = (position / block_size_m).array().floor().cast<int>();
  *voxel_idx =
      ((position - block_size_m * block_idx->cast<float>()) * voxel_size_m_inv)
          .array()
          .cast<int>()
          .min(kVoxelsPerSideMinusOne);
}

Vector3f getPositionFromBlockIndexAndVoxelIndex(const float block_size_m,
                                                const Index3D& block_index,
                                                const Index3D& voxel_index) {
  const float voxel_size_m = blockSizeToVoxelSize(block_size_m);
  return Vector3f(block_size_m * block_index.cast<float>() +
                  voxel_size_m * voxel_index.cast<float>());
}

Vector3f getPositionFromBlockIndex(const float block_size_m,
                                   const Index3D& block_index) {
  // This is pretty trivial, huh.
  return Vector3f(block_size_m * block_index.cast<float>());
}

Vector3f getCenterPositionFromBlockIndex(const float block_size_m,
                                         const Index3D& block_index) {
  // This is pretty trivial, huh.
  return Vector3f(block_size_m * (block_index.cast<float>().array() + 0.5f));
}

Vector3f getCenterPositionFromBlockIndexAndVoxelIndex(
    const float block_size_m, const Index3D& block_index,
    const Index3D& voxel_index) {
  constexpr float kHalfVoxelsPerSideInv =
      0.5f / VoxelBlock<bool>::kVoxelsPerSide;
  const float half_voxel_size_m = block_size_m * kHalfVoxelsPerSideInv;

  return getPositionFromBlockIndexAndVoxelIndex(block_size_m, block_index,
                                                voxel_index) +
         Vector3f(half_voxel_size_m, half_voxel_size_m, half_voxel_size_m);
}

// 2D Indexing
Vector2f get2DPositionFromBlockIndexAndVoxelIndex(const float block_size_m,
                                                  const Index2D& block_index,
                                                  const Index2D& voxel_index) {
  const float voxel_size_m = blockSizeToVoxelSize(block_size_m);
  return Vector2f(block_size_m * block_index.cast<float>() +
                  voxel_size_m * voxel_index.cast<float>());
}

Vector2f getCenter2DPositionFromBlockIndexAndVoxelIndex(
    const float block_size_m, const Index2D& block_2d_index,
    const Index2D& voxel_2d_index) {
  constexpr float kHalfVoxelsPerSideInv =
      0.5f / VoxelBlock<bool>::kVoxelsPerSide;
  const float half_voxel_size_m = block_size_m * kHalfVoxelsPerSideInv;

  return get2DPositionFromBlockIndexAndVoxelIndex(block_size_m, block_2d_index,
                                                  voxel_2d_index) +
         Vector2f(half_voxel_size_m, half_voxel_size_m);
}

// 1D indexing
std::pair<int, int> getBlockAndVoxelIndexFrom1DPositionInLayer(
    const float block_size_m, float p) {
  constexpr int kVoxelsPerSideMinusOne = VoxelBlock<bool>::kVoxelsPerSide - 1;
  const float voxel_size_m_inv = 1.0 / blockSizeToVoxelSize(block_size_m);
  const int block_idx = static_cast<int>(std::floor(p / block_size_m));
  const int voxel_idx = std::min(
      static_cast<int>((p - block_size_m * static_cast<float>(block_idx)) *
                       voxel_size_m_inv),
      kVoxelsPerSideMinusOne);
  return {block_idx, voxel_idx};
}

}  // namespace nvblox
