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

namespace nvblox {

ColumnBounds::ColumnBounds(int max_voxel_idx_z, int min_voxel_idx_z,
                           int max_block_idx_z, int min_block_idx_z)
    : max_voxel_idx_z_(max_voxel_idx_z),  // NOLINT
      min_voxel_idx_z_(min_voxel_idx_z),  // NOLINT
      max_block_idx_z_(max_block_idx_z),  // NOLINT
      min_block_idx_z_(min_block_idx_z) {
  assert(max_block_idx_z_ >= min_block_idx_z_);
  // If we're in the same block, assert (max_voxel_idx_z_ >= min_voxel_idx_z_)
  assert((max_voxel_idx_z_ >= min_voxel_idx_z_) ||
         (max_block_idx_z_ != min_block_idx_z_));
}

__host__ __device__ std::pair<int, int> ColumnBounds::getMinAndMaxVoxelZIndex(
    int block_idx_z) const {
  // Default is full voxel range
  int min_voxel_idx_z_out = 0;
  int max_voxel_idx_z_out = VoxelBlock<bool>::kVoxelsPerSide - 1;
  // If on the extreme blocks, assume their voxel bounds.
  if (block_idx_z == min_block_idx_z_) {
    min_voxel_idx_z_out = min_voxel_idx_z_;
  }
  if (block_idx_z == max_block_idx_z_) {
    max_voxel_idx_z_out = max_voxel_idx_z_;
  }
  // If block index outside of range, skip iteration by reversing voxel range.
  if (block_idx_z < min_block_idx_z_ || block_idx_z > max_block_idx_z_) {
    min_voxel_idx_z_out = 1;
    max_voxel_idx_z_out = 0;
  }
  assert(min_voxel_idx_z_out >= 0);
  assert(max_voxel_idx_z_out >= 0);
  assert(min_voxel_idx_z_out < TsdfBlock::kNumVoxels);
  assert(max_voxel_idx_z_out < TsdfBlock::kNumVoxels);
  return {min_voxel_idx_z_out, max_voxel_idx_z_out};
}

__host__ __device__ bool ColumnBounds::isBlockIdxInRange(
    const int block_idx_z) const {
  return (block_idx_z >= min_block_idx_z_ && block_idx_z <= max_block_idx_z_);
}

ConstantZColumnBoundsGetter::ConstantZColumnBoundsGetter(float min_z,
                                                         float max_z,
                                                         float block_size) {
  // Find the minimum z-index that should be included in the slice
  std::tie(min_block_idx_z_, min_voxel_idx_z_) =
      getBlockAndVoxelIndexFrom1DPositionInLayer(block_size, min_z);
  std::tie(max_block_idx_z_, max_voxel_idx_z_) =
      getBlockAndVoxelIndexFrom1DPositionInLayer(block_size, max_z);
  CHECK_GE(max_block_idx_z_, min_block_idx_z_);
  CHECK_LT(max_voxel_idx_z_, TsdfBlock::kVoxelsPerSide);
  CHECK_GE(min_voxel_idx_z_, 0);
}

__host__ __device__ ColumnBounds ConstantZColumnBoundsGetter::getColumnBounds(
    const Index2D&, const Index2D&) const {
  return ColumnBounds(max_voxel_idx_z_, min_voxel_idx_z_, max_block_idx_z_,
                      min_block_idx_z_);
}

__host__ PlanarSliceColumnBoundsGetter::PlanarSliceColumnBoundsGetter(
    const Plane& low_plane,            // NOLINT
    float slice_height_above_plane_m,  // NOLINT
    float slice_height_thickness_m,    // NOLINT
    float block_size_m)
    : low_plane_(low_plane),
      slice_height_above_plane_m_(slice_height_above_plane_m),
      slice_height_thickness_m_(slice_height_thickness_m),
      block_size_m_(block_size_m) {
  // Require that plane is not vertical (because this would make this slice
  // ill-defined).
  constexpr float kEps = 1.0e-4;
  CHECK_GT(fabs(low_plane_.normal().z()), kEps)
      << "PlanarSliceColumnBoundsGetter requires that the input plane not be "
         "close to vertical.";
  CHECK_GE(slice_height_above_plane_m, 0);
  CHECK_GE(slice_height_thickness_m, 0);
  // The maximum num of blocks spanned by the thickness of the slice.
  // For example a width of 1.1*block_size can span 3 blocks if the span falls
  // centered on a block.
  num_blocks_in_vertical_column_ =
      static_cast<int>(ceilf(slice_height_thickness_m_ / block_size_m_)) + 1;
  CHECK_GT(num_blocks_in_vertical_column_, 0);
}

__host__ __device__ ColumnBounds PlanarSliceColumnBoundsGetter::getColumnBounds(
    const Index2D& block_xy_idx, const Index2D& voxel_xy_idx) const {
  // Get the 2D coordinates of this voxel in the XY plane
  const Vector2f p_xy = getCenter2DPositionFromBlockIndexAndVoxelIndex(
      block_size_m_, block_xy_idx, voxel_xy_idx);

  // Get the plane height at this location
  const float plane_height_m = low_plane_.getHeightAtXY(p_xy);

  // Get the height of the low and high slices.
  const float min_slice_height_m = plane_height_m + slice_height_above_plane_m_;
  const float max_slice_height_m =
      min_slice_height_m + slice_height_thickness_m_;
  assert(min_slice_height_m >= plane_height_m);
  assert(max_slice_height_m >= plane_height_m);

  // Single axis indexing
  const auto [min_block_idx_z, min_voxel_idx_z] =
      getBlockAndVoxelIndexFrom1DPositionInLayer(block_size_m_,
                                                 min_slice_height_m);
  const auto [max_block_idx_z, max_voxel_idx_z] =
      getBlockAndVoxelIndexFrom1DPositionInLayer(block_size_m_,
                                                 max_slice_height_m);

  return ColumnBounds(max_voxel_idx_z, min_voxel_idx_z,  // NOLINT
                      max_block_idx_z, min_block_idx_z);
}

}  // namespace nvblox
