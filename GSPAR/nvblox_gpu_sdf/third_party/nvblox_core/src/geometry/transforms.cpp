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
#include "nvblox/core/types.h"

namespace nvblox {

bool arePosesClose(const Transform& T_A_B1, const Transform& T_A_B2,
                   const float translation_tolerance_m,
                   const float angular_tolerance_deg) {
  // Check that the cameras have the same extrinsics
  const Transform T_B1_B2 = T_A_B1.inverse() * T_A_B2;
  if (T_B1_B2.translation().norm() > translation_tolerance_m) {
    return false;
  }
  const float angle_between_cameras_rad =
      Eigen::AngleAxisf(T_B1_B2.rotation()).angle();
  const float angle_between_cameras_deg =
      angle_between_cameras_rad * 180.0f / M_PI;
  if (std::abs(angle_between_cameras_deg) > angular_tolerance_deg) {
    return false;
  }
  return true;
}

}  // namespace nvblox
