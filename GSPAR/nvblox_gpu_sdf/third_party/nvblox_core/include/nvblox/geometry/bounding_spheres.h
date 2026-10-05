/*
Copyright 2022-2024 NVIDIA CORPORATION

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

class BoundingSphere {
 public:
  BoundingSphere() : center_(Vector3f::Zero()), radius_(0.f) {}
  BoundingSphere(const Vector3f& center, float radius)
      : center_(center), radius_(radius) {}
  ~BoundingSphere() = default;

  __host__ __device__ bool contains(const Vector3f& point) const {
    float distance_to_center = (center_ - point).norm();
    return distance_to_center <= radius_;
  };

  Vector3f center() const { return center_; }
  float radius() const { return radius_; }

 protected:
  Vector3f center_;
  float radius_;
};

/// Check whether the block is partially or fully within a radius around the
/// center.
/// @param block_index The index of the block.
/// @param block_size Metric size of the block.
/// @param center The center from which we measure the radius.
/// @param radius The radius in meters.
/// @return Whether the block is partially or fully within the radius.
bool isBlockWithinRadius(const Index3D& block_index, float block_size,
                         const Vector3f& center, float radius);

/// Check whether the block is fully outside a radius around the
/// center.
/// @param block_index The index of the block.
/// @param block_size Metric size of the block.
/// @param center The center from which we measure the radius.
/// @param radius The radius in meters.
/// @return Whether the block is fully outside the radius.
bool isBlockOutsideRadius(const Index3D& block_index, float block_size,
                          const Vector3f& center, float radius);

/// Get all of the blocks that are partially or fully within a radius around the
/// center.
/// @param input_blocks The selection of blocks that should be tested.
/// @param block_size Metric size of the block.
/// @param center The center from which we measure the radius.
/// @param radius The radius in meters.
/// @return All block indices of the input blocks that are partially or fully
/// within the radius.
std::vector<Index3D> getBlocksWithinRadius(
    const std::vector<Index3D>& input_blocks, float block_size,
    const Vector3f& center, float radius);

/// Get all of the blocks that are fully outside a radius around the
/// center.
/// @param input_blocks The selection of blocks that should be tested.
/// @param block_size Metric size of the block.
/// @param center The center from which we measure the radius.
/// @param radius The radius in meters.
/// @return All block indices of the input blocks that are fully
/// outside the radius.
std::vector<Index3D> getBlocksOutsideRadius(
    const std::vector<Index3D>& input_blocks, float block_size,
    const Vector3f& center, float radius);

/// Get all of the blocks that are within a radius around the
/// AABB.
/// @param input_blocks The selection of blocks that should be tested.
/// @param block_size Metric size of the block.
/// @param aabb The bounding box (AABB).
/// @param radius The radius in meters.
/// @return All block indices of the input blocks that are within a radius
/// around the AABB.
std::vector<Index3D> getBlocksWithinRadiusOfAABB(
    const std::vector<Index3D>& input_blocks, float block_size,
    const AxisAlignedBoundingBox& aabb, float radius);

}  // namespace nvblox
