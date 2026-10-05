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

#include <utility>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"

namespace nvblox {

// Structs which describe slicing of the TSDF/occupancy layer.

/// Describes a z-parallel slice (in the Layer frame).
struct ConstantZSliceDescription {
  /// The height (in meters) of the bottom slice plane in the Layer frame.
  float z_min_m;
  /// The height (in meters) of the top slice plane in the Layer frame.
  float z_max_m;
  /// The height (in meters) of the desired output for the ESDF slice.
  float z_output_m;
};

/// Describes a planar slice (in the Layer frame).
/// In contrast to the constant-z slice there are no limitations on the
/// plane's orientation.
/// The slide in this case consists of a (parallel) low-side and high-side
/// plane. We look for site-voxels between these two planes. The low-side
/// plane is offset from the "ground_plane" by "slice_height_above_plane_m"
/// and the high-side plane is offset by "slice_height_thickness_m" above the
/// low-side plane in z.
struct PlanarSliceDescription {
  /// The plane defining the ground. The low-side slice plane is parallel with
  /// this, the ground-plane but offset.
  const Plane ground_plane;
  /// The height of the low-side slice above the ground-plane (in z).
  float slice_height_above_plane_m;
  /// The thickness of the slice in the z dimension. In other words, the
  /// distance (in z) between the low-side slice plane and the high-side slice
  /// plane.
  float slice_height_thickness_m;
  /// The height (in meters) of the desired output for the ESDF slice.
  float z_output_m;
};

/// Defines the bounds of a slice within a voxel column, i.e. for a single 2D
/// location in the x-y plane.
class ColumnBounds {
 public:
  /// Defines the bounds within a voxel column.
  /// @param max_voxel_idx_z The max voxel idx in the max block.
  /// @param min_voxel_idx_z The min voxel idx in the min block.
  /// @param max_block_idx_z The max block index in this column.
  /// @param min_block_idx_z The min block index in this column.
  __host__ __device__ ColumnBounds(int max_voxel_idx_z, int min_voxel_idx_z,
                                   int max_block_idx_z, int min_block_idx_z);

  /// For a given block index, return the min and max voxel indices we should
  /// span.
  /// @param block_idx_z The block_index for which we want the voxel_index span.
  /// @return A pair containing the [min, max] voxel indices. Note that this is
  /// inclusive. The indicated min and max voxels should be processed.
  __host__ __device__ inline std::pair<int, int> getMinAndMaxVoxelZIndex(
      int block_idx_z) const;

  /// Returns true if the z block_idx is within the slice range.
  /// @param block_idx_z The z-idx of the block.
  /// @return True if in range.
  __host__ __device__ inline bool isBlockIdxInRange(
      const int block_idx_z) const;

  /// The max voxel idx in the max block.
  __host__ __device__ int max_voxel_idx_z() const { return max_voxel_idx_z_; }
  /// The min voxel idx in the min block.
  __host__ __device__ int min_voxel_idx_z() const { return min_voxel_idx_z_; }
  /// The max block index in this column.
  __host__ __device__ int max_block_idx_z() const { return max_block_idx_z_; }
  /// The min block index in this column.
  __host__ __device__ int min_block_idx_z() const { return min_block_idx_z_; }

 protected:
  /// Max voxel index in the top block
  int max_voxel_idx_z_;
  /// Min voxel index in the bottom block
  int min_voxel_idx_z_;
  /// Max block index
  int max_block_idx_z_;
  /// Min block index
  int min_block_idx_z_;
};

/// Interface for classes which are used to do slicing. The critical method is
/// getColumnBounds() which returns the min/max block and voxel indices which
/// define the slice for a particular position in the x,y plane.
class ColumnBoundsGetterInterface {
 public:
  ColumnBoundsGetterInterface() = default;
  virtual ~ColumnBoundsGetterInterface() = default;

  /// Returns the number of VoxelBlocks in a column of the slice.
  /// @return Number of voxel blocks.
  virtual int num_blocks_in_vertical_column() const = 0;

  /// Gets the z-bounds for the requested column of voxels.
  /// @param block_idx Block index in the x,y plane.
  /// @param voxel_idx Voxel index in the x,y plane.
  /// @return The bounds for voxel column.
  __host__ __device__ virtual ColumnBounds getColumnBounds(
      const Index2D& block_xy_idx, const Index2D& voxel_xy_idx) const = 0;
};

/// This functor returns the block and voxel indices of bottom and top of the
/// slice.
class ConstantZColumnBoundsGetter : public ColumnBoundsGetterInterface {
 public:
  /// Creates an ConstantZColumnBoundsGetter which returns the voxel_index
  /// bounds of the slice.
  /// @param min_z The height of the lower slice plane (in layer frame).
  /// @param max_z The height of the upper slice plane (in layer frame).
  /// @param block_size The size of a (Voxel)Block.
  __host__ ConstantZColumnBoundsGetter(float min_z, float max_z,
                                       float block_size_m);

  /// Returns the number of vertical blocks between the slice bounds.
  /// @return The number of vertical blocks.
  int num_blocks_in_vertical_column() const {
    return max_block_idx_z_ - min_block_idx_z_ + 1;
  }

  /// Gets the z-bounds for the requested column of voxels.
  /// @param block_idx Block index in the x,y plane.
  /// @param voxel_idx Voxel index in the x,y plane.
  /// @return The bounds for voxel column.
  __host__ __device__ ColumnBounds getColumnBounds(
      const Index2D& block_xy_idx, const Index2D& voxel_xy_idx) const;

 protected:
  /// Max voxel index in the top block
  int max_voxel_idx_z_;
  /// Min voxel index in the bottom block
  int min_voxel_idx_z_;
  /// Max block index
  int max_block_idx_z_;
  /// Min block index
  int min_block_idx_z_;
};

/// A column bounds getter that uses a plane to determine the slice. The slice
/// is performed between a height above a ground plane, and with a specified
/// thickness.
struct PlanarSliceColumnBoundsGetter : public ColumnBoundsGetterInterface {
  /// A column bounds getter that uses a plane to determine the slice.
  /// Requires: The low-plane not be very close to vertical, otherwise we fail.
  /// @param low_plane The (ground) plane.
  /// @param slice_height_above_plane_m Start height for the slice above the
  /// ground plane.
  /// @param slice_height_thickness_m The thickness of the slice (in z).
  /// @param block_size_m Size of a VoxelBlock
  __host__ PlanarSliceColumnBoundsGetter(
      const Plane& low_plane,            // NOLINT
      float slice_height_above_plane_m,  // NOLINT
      float slice_height_thickness_m,    // NOLINT
      float block_size_m);

  /// Returns the number of vertical blocks between the slice bounds.
  /// @return The number of vertical blocks.
  int num_blocks_in_vertical_column() const {
    return num_blocks_in_vertical_column_;
  }

  /// Gets the z-bounds for the requested column of voxels.
  /// @param block_idx Block index in the x,y plane.
  /// @param voxel_idx Voxel index in the x,y plane.
  /// @return The bounds for voxel column.
  __host__ __device__ ColumnBounds getColumnBounds(
      const Index2D& block_xy_idx, const Index2D& voxel_xy_idx) const;

 protected:
  /// The plane that defines the lower slice bound.
  Plane low_plane_;

  /// The height of the slice in meters.
  float slice_height_above_plane_m_;

  /// The distance in z between the two slice planes.
  float slice_height_thickness_m_;

  /// Number of blocks in a column
  int num_blocks_in_vertical_column_;

  /// The size of blocks in the map.
  float block_size_m_;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/cuda/impl/esdf_integrator_slicing_impl.cuh"
