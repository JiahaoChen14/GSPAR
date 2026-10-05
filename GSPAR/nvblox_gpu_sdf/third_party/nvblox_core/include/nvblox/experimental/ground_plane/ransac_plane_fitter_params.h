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

enum class RansacType {
  // Uses the default Ransac where the model with the most inliers
  // (points within a defined distance threshold) after a set number of
  // iterations is selected as the best-fitting plane.
  kRansac,
  // Uses the M-Estimator Sample Consensus MSAC to compute
  // the cost of a sample fit as the rescending M-Esimator as described in
  // https://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf
  kMSAC,
};

constexpr Param<float>::Description kRansacDistanceThresholdMDesc{
    "ransac_distance_threshold_m", 0.2,
    "RANSAC Parameter: The maximum distance in m a point can be from a plane "
    "to be considered an inlier."};

constexpr Param<RansacType>::Description kRansacTypeDesc{
    "ransac_type", RansacType::kMSAC,
    "0: Uses the default Ransac where the model with the most inliers"
    "(points within a defined distance threshold) after a set number of"
    " iterations is selected as the best-fitting plane."
    "1: Uses the M-Estimator Sample Consensus MSAC to compute"
    "the cost of a sample fit as the rescending M-Esimator as described in"
    "https://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf"};

constexpr Param<int>::Description kNumRansacIterationsDesc{
    "num_ransac_iterations", 1000, "Iterations to run the RANSAC algorithm."};

struct RansacPlaneFitterParams {
  Param<float> ransac_distance_threshold_m{kRansacDistanceThresholdMDesc};
  Param<RansacType> ransac_type{kRansacTypeDesc};
  Param<int> num_ransac_iterations{kNumRansacIterationsDesc};
};
}  // namespace nvblox
