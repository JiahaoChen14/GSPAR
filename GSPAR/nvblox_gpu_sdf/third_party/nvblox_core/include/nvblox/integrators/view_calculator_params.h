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

#include "nvblox/geometry/workspace_bounds.h"
#include "nvblox/utils/params.h"

namespace nvblox {

constexpr Param<int>::Description kRaycastSubsamplingFactorDesc{
    "raycast_subsampling_factor", 4,
    "The rate at which we subsample pixels to raycast. Note that we always "
    "raycast the extremes of the frame, no matter the subsample rate."};

constexpr Param<WorkspaceBoundsType>::Description kWorkspaceBoundsTypeDesc{
    "workspace_bounds_type", WorkspaceBoundsType::kUnbounded,
    "The type of bounds we limit the workspace with (unbounded, height bounds "
    "or bounding box)."};

constexpr Param<float>::Description kWorkspaceBoundsMinHeightDesc{
    "workspace_bounds_min_height_m", 0.0f,
    "The minimal height of the workspace bounds."};

constexpr Param<float>::Description kWorkspaceBoundsMaxHeightDesc{
    "workspace_bounds_max_height_m", 1.0f,
    "The maximal height of the workspace bounds."};

constexpr Param<float>::Description kWorkspaceBoundsMinCornerXDesc{
    "workspace_bounds_min_corner_x_m", 0.0f,
    "The x-component of the minimal corner of the workspace bounds. Only used "
    "if workspace_bounds_type:=bounding_box."};

constexpr Param<float>::Description kWorkspaceBoundsMaxCornerXDesc{
    "workspace_bounds_max_corner_x_m", 0.0f,
    "The x-component of the  maximal corner of the workspace bounds. Only used "
    "if workspace_bounds_type:=bounding_box."};

constexpr Param<float>::Description kWorkspaceBoundsMinCornerYDesc{
    "workspace_bounds_min_corner_y_m", 2.0f,
    "The y-component of the  minimal corner of the workspace bounds. Only used "
    "if workspace_bounds_type:=bounding_box."};

constexpr Param<float>::Description kWorkspaceBoundsMaxCornerYDesc{
    "workspace_bounds_max_corner_y_m", 2.0f,
    "The y-component of the  maximal corner of the workspace bounds. Only used "
    "if workspace_bounds_type:=bounding_box."};

struct ViewCalculatorParams {
  Param<int> raycast_subsampling_factor{kRaycastSubsamplingFactorDesc};
  Param<WorkspaceBoundsType> workspace_bounds_type{kWorkspaceBoundsTypeDesc};
  Param<float> workspace_bounds_min_height_m{kWorkspaceBoundsMinHeightDesc};
  Param<float> workspace_bounds_max_height_m{kWorkspaceBoundsMaxHeightDesc};
  Param<float> workspace_bounds_min_corner_x_m{kWorkspaceBoundsMinCornerXDesc};
  Param<float> workspace_bounds_max_corner_x_m{kWorkspaceBoundsMaxCornerXDesc};
  Param<float> workspace_bounds_min_corner_y_m{kWorkspaceBoundsMinCornerYDesc};
  Param<float> workspace_bounds_max_corner_y_m{kWorkspaceBoundsMaxCornerYDesc};
};

}  // namespace nvblox
