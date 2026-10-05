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
#include "nvblox/geometry/workspace_bounds.h"

namespace nvblox {

bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const WorkspaceBoundsType bounds_type,
                          const Vector3f& workspace_bounds_min_corner,
                          const Vector3f& workspace_bounds_max_corner,
                          AxisAlignedBoundingBox* output_aabb) {
  switch (bounds_type) {
    case WorkspaceBoundsType::kUnbounded:
      *output_aabb = input_aabb;
      return !output_aabb->isEmpty();
    case WorkspaceBoundsType::kHeightBounds:
      return applyWorkspaceBounds(input_aabb, workspace_bounds_min_corner.z(),
                                  workspace_bounds_max_corner.z(), output_aabb);
    case WorkspaceBoundsType::kBoundingBox:
      return applyWorkspaceBounds(input_aabb, workspace_bounds_min_corner,
                                  workspace_bounds_max_corner, output_aabb);
    default:
      LOG(FATAL) << "WorkspaceBoundsType not implemented: " << bounds_type;
      break;
  }
}

bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const Vector3f& workspace_bounds_min_corner,
                          const Vector3f& workspace_bounds_max_corner,
                          AxisAlignedBoundingBox* output_aabb) {
  AxisAlignedBoundingBox workspace_aabb(workspace_bounds_min_corner,
                                        workspace_bounds_max_corner);
  *output_aabb = workspace_aabb.intersection(input_aabb);
  return !output_aabb->isEmpty();
}

bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const float workspace_bounds_min_height,
                          const float workspace_bounds_max_height,
                          AxisAlignedBoundingBox* output_aabb) {
  *output_aabb = input_aabb;
  output_aabb->min().z() =
      std::max(input_aabb.min().z(), workspace_bounds_min_height);
  output_aabb->max().z() =
      std::min(input_aabb.max().z(), workspace_bounds_max_height);
  return !output_aabb->isEmpty();
}

}  // namespace nvblox
