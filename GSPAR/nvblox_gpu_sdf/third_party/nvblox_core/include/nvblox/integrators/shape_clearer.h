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

#include "nvblox/geometry/bounding_shape.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// Class to clear parts of a layer.
/// The ShapeClearer accepts bounding shapes (e.g. spheres or bounding boxes)
/// and clears the voxels that are contained inside these shapes.
/// @tparam LayerType The type of the layer.
template <class LayerType>
class ShapeClearer {
 public:
  using BlockType = typename LayerType::BlockType;
  using VoxelType = typename BlockType::VoxelType;
  ShapeClearer();
  ShapeClearer(std::shared_ptr<CudaStream> cuda_stream);
  ~ShapeClearer() = default;

  /// @brief Clearing the bounding shapes in the layer.
  /// @param bounding_shapes The bounding shapes that define which voxels to
  /// clear.
  /// @param layer_ptr The layer we want to clear.
  /// @return The block indices that have been updated during the clearing.
  std::vector<Index3D> clear(const std::vector<BoundingShape>& bounding_shapes,
                             LayerType* layer_ptr);

 private:
  // Host and device buffers.
  host_vector<Index3D> block_indices_host_;
  host_vector<BlockType*> block_ptrs_host_;
  host_vector<BoundingShape> shapes_to_clear_host_;
  device_vector<Index3D> block_indices_device_;
  device_vector<BlockType*> block_ptrs_device_;
  device_vector<BoundingShape> shapes_to_clear_device_;

  // CUDA stream to process integration on.
  std::shared_ptr<CudaStream> cuda_stream_;
};

/// Typedefs of ShapeClearer.
typedef ShapeClearer<TsdfLayer> TsdfShapeClearer;
typedef ShapeClearer<OccupancyLayer> OccupancyShapeClearer;
typedef ShapeClearer<ColorLayer> ColorShapeClearer;

}  // namespace nvblox
