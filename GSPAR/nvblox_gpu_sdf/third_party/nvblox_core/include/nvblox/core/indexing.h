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

#include "nvblox/core/types.h"

namespace nvblox {

__host__ __device__ inline constexpr float voxelSizeToBlockSize(
    const float voxel_size_m);
__host__ __device__ inline constexpr float blockSizeToVoxelSize(
    const float block_size_m);

__host__ __device__ inline Index3D getBlockIndexFromPositionInLayer(
    const float block_size_m, const Vector3f& position);

__host__ __device__ inline void getBlockAndVoxelIndexFromPositionInLayer(
    const float block_size_m, const Vector3f& position, Index3D* block_idx,
    Index3D* voxel_idx);

/// Gets the position of the minimum corner (i.e., the smallest towards negative
/// infinity of each axis).
__host__ __device__ inline Vector3f getPositionFromBlockIndexAndVoxelIndex(
    const float block_size_m, const Index3D& block_index,
    const Index3D& voxel_index);

__host__ __device__ inline Vector3f getPositionFromBlockIndex(
    const float block_size_m, const Index3D& block_index);

/// Gets the CENTER of the voxel.
__host__ __device__ inline Vector3f
getCenterPositionFromBlockIndexAndVoxelIndex(const float block_size_m,
                                             const Index3D& block_index,
                                             const Index3D& voxel_index);

__host__ __device__ inline Vector3f getCenterPositionFromBlockIndex(
    const float block_size_m, const Index3D& block_index);

// 2D Indexing
__host__ __device__ inline Vector2f get2DPositionFromBlockIndexAndVoxelIndex(
    const float block_size_m, const Index2D& block_index,
    const Index2D& voxel_index);

__host__ __device__ inline Vector2f
getCenter2DPositionFromBlockIndexAndVoxelIndex(const float block_size_m,
                                               const Index2D& block_2d_index,
                                               const Index2D& voxel_2d_index);

// 1D indexing
__host__ __device__ inline std::pair<int, int>
getBlockAndVoxelIndexFrom1DPositionInLayer(const float block_size_m, float p);

}  // namespace nvblox

#include "nvblox/core/internal/impl/indexing_impl.h"
