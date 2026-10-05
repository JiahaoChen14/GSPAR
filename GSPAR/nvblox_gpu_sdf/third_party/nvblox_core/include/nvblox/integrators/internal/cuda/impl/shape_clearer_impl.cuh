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
#include "nvblox/integrators/shape_clearer.h"

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

__device__ inline void clearVoxel(TsdfVoxel* voxel_ptr) {
  voxel_ptr->distance = 0.f;
  voxel_ptr->weight = 0.f;
}

__device__ inline void clearVoxel(OccupancyVoxel* voxel_ptr) {
  voxel_ptr->log_odds = 0.f;
}

__device__ inline void clearVoxel(ColorVoxel* voxel_ptr) {
  voxel_ptr->color = Color::Gray();
  voxel_ptr->weight = 0.f;
}

template <typename VoxelType>
__global__ void clearShapesKernel(const Index3D* block_indices_device_ptr,
                                  const float block_size,
                                  const BoundingShape* shape_list,
                                  const int shape_list_size,
                                  VoxelBlock<VoxelType>** block_device_ptrs) {
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

  // Voxel center point
  const Vector3f p_voxel_center = getCenterPositionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);

  for (size_t i = 0; i < shape_list_size; i++) {
    if (shape_list[i].contains(p_voxel_center)) {
      VoxelType* voxel_ptr =
          &(block_device_ptrs[blockIdx.x]
                ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);
      clearVoxel(voxel_ptr);
    }
  }
}

template <typename LayerType>
ShapeClearer<LayerType>::ShapeClearer()
    : ShapeClearer(std::make_shared<CudaStreamOwning>()) {}

template <typename LayerType>
ShapeClearer<LayerType>::ShapeClearer(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

template <typename LayerType>
std::vector<Index3D> ShapeClearer<LayerType>::clear(
    const std::vector<BoundingShape>& bounding_shapes, LayerType* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  const float block_size = layer_ptr->block_size();

  // Predicate that returns true for blocks that are partially or fully
  // contained by on of the AABBs.
  auto predicate = [&bounding_shapes, &block_size](const Index3D& index) {
    for (const auto& shape_variant : bounding_shapes) {
      if (shape_variant.touchesBlock(index, block_size)) {
        return true;
      }
    }
    return false;
  };
  const std::vector<Index3D> block_indices =
      layer_ptr->getBlockIndicesIf(predicate);
  const int num_blocks = block_indices.size();

  std::vector<VoxelBlock<VoxelType>*> block_ptrs;
  block_ptrs.reserve(num_blocks);
  for (const auto& index : block_indices) {
    block_ptrs.push_back(layer_ptr->getBlockAtIndex(index).get());
  }

  if (num_blocks == 0) {
    // No blocks inside bounding boxes, nothing to do here.
    return std::vector<Index3D>();
  }
  CHECK_EQ(block_ptrs.size(), block_indices.size());

  expandBuffersIfRequired(block_indices.size(), *cuda_stream_,
                          &block_indices_host_, &block_ptrs_host_,
                          &block_indices_device_, &block_ptrs_device_);

  // Get the aabbs on host and copy them to device
  shapes_to_clear_host_.copyFromAsync(bounding_shapes, *cuda_stream_);
  shapes_to_clear_device_.copyFromAsync(shapes_to_clear_host_, *cuda_stream_);

  // Get the block indices on host and copy them to device
  block_indices_host_.copyFromAsync(block_indices, *cuda_stream_);
  block_indices_device_.copyFromAsync(block_indices_host_, *cuda_stream_);

  // Get the block pointers on host and copy them to device
  block_ptrs_host_.copyFromAsync(block_ptrs, *cuda_stream_);
  block_ptrs_device_.copyFromAsync(block_ptrs_host_, *cuda_stream_);

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  clearShapesKernel<<<num_blocks, kThreadsPerBlock, 0, *cuda_stream_>>>(
      block_indices_device_.data(),    // NOLINT
      layer_ptr->block_size(),         // NOLINT
      shapes_to_clear_device_.data(),  // NOLINT
      shapes_to_clear_device_.size(),  // NOLINT
      block_ptrs_device_.data());      // NOLINT
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  return block_indices;
}

}  // namespace nvblox
