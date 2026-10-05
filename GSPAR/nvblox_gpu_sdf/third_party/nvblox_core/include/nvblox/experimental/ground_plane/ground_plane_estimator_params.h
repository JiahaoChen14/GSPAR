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

#include "nvblox/utils/params.h"

namespace nvblox {

constexpr Param<float>::Description kGroundPointsCandidatesMinZMDesc{
    "ground_points_candidates_min_z_m", -0.1,
    "Minimum height in meters for tsdf zero crossings to be "
    "considered candidates for ground points."};

constexpr Param<float>::Description kGroundPointsCandidatesMaxZMDesc{
    "ground_points_candidates_max_z_m", 0.15,
    "Maximum height in meters for tsdf zero crossings to be "
    "considered candidates for ground points."};

struct GroundPlaneEstimatorParams {
  Param<float> ground_points_candidates_min_z_m{
      kGroundPointsCandidatesMinZMDesc};
  Param<float> ground_points_candidates_max_z_m{
      kGroundPointsCandidatesMaxZMDesc};
};

}  // namespace nvblox