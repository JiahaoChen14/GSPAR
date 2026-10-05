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

#include "nvblox/core/types.h"

namespace nvblox {

/// @brief Enum defining the workspace bounds type (limiting the space in which
/// we update the map).
enum class WorkspaceBoundsType { kUnbounded, kHeightBounds, kBoundingBox };

/// @brief Apply the workspace bounds on an input bounding box.
/// @param input_aabb The input bounding box on which we apply the workspace
/// bounds.
/// @param bounds_type The type of workspace bounds we apply.
/// @param workspace_bounds_min_corner The minimal corner of the workspace
/// bounds.
/// @param workspace_bounds_max_corner  The maximal corner of the workspace
/// bounds.
/// @param output_aabb The resulting AABB of the intersection between the input
/// bounding box and the workspace bounds.
/// @return Boolean whether the resulting AABB is non-empty.
bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const WorkspaceBoundsType bounds_type,
                          const Vector3f& workspace_bounds_min_corner,
                          const Vector3f& workspace_bounds_max_corner,
                          AxisAlignedBoundingBox* output_aabb);

/// @brief Apply a workspace bounding box on an input bounding box.
/// @param input_aabb The input bounding box on which we apply the workspace
/// bounding box.
/// @param workspace_bounds_min_corner The minimal corner of the workspace
/// bounding box.
/// @param workspace_bounds_max_corner  The maximal corner of the workspace
/// bounding box.
/// @param output_aabb The resulting AABB of the intersection between the input
/// bounding box and the workspace bounding box.
/// @return Boolean whether the resulting AABB is non-empty.
bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const Vector3f& workspace_bounds_min_corner,
                          const Vector3f& workspace_bounds_max_corner,
                          AxisAlignedBoundingBox* output_aabb);

/// @brief Apply a workspace height bounds on an input bounding box.
/// @param input_aabb The input bounding box on which we apply the workspace
/// height bounds.
/// @param workspace_bounds_min_height The minimal height of the workspace
/// height bounds.
/// @param workspace_bounds_max_height  The maximal height of the workspace
/// height bounds.
/// @param output_aabb The resulting AABB of the intersection between the input
/// bounding box and the workspace height bounds.
/// @return Boolean whether the resulting AABB is non-empty.
bool applyWorkspaceBounds(const AxisAlignedBoundingBox& input_aabb,
                          const float workspace_bounds_min_height,
                          const float workspace_bounds_max_height,
                          AxisAlignedBoundingBox* output_aabb);

inline std::ostream& operator<<(std::ostream& os,
                                const WorkspaceBoundsType& bounds_type) {
  switch (bounds_type) {
    case WorkspaceBoundsType::kUnbounded:
      os << "kUnbounded";
      break;
    case WorkspaceBoundsType::kHeightBounds:
      os << "kHeightBounds";
      break;
    case WorkspaceBoundsType::kBoundingBox:
      os << "kBoundingBox";
      break;
    default:
      break;
  }
  return os;
}

inline std::string to_string(const WorkspaceBoundsType& bounds_type) {
  std::ostringstream ss;
  ss << bounds_type;
  return ss.str();
}

}  // namespace nvblox
