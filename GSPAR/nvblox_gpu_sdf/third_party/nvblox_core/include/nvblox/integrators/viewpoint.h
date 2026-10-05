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

#include <optional>

#include "nvblox/core/types.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// A structure describing a view of the scene.
struct ViewBasedInclusionData {
  ViewBasedInclusionData() = delete;
  explicit ViewBasedInclusionData(
      Transform _T_L_C, Camera _camera,
      std::optional<const DepthImage*> _depth_image = std::nullopt,
      std::optional<float> _max_view_distance_m = std::nullopt,
      std::optional<float> _truncation_distance_m = std::nullopt);
  ~ViewBasedInclusionData() = default;

  /// The pose of the camera for view-based decay-exclusion.
  Transform T_L_C;
  /// The intrinsics of the camera for view-based decay-exclusion.
  Camera camera;
  /// The depth image tested for valid depth during view-based decay-exclusion.
  std::optional<const DepthImage*> depth_image;
  /// The maximum depth at which a voxel is considered in view. If these are not
  /// provided the max distance is infinite.
  std::optional<float> max_view_distance_m;
  /// truncation_distance_m behind the depth measurement is considered occluded
  /// and will be decayed. If this is not provided, we do not do occlusion
  /// testing.
  std::optional<float> truncation_distance_m;
};

}  // namespace nvblox
