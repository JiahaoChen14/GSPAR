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

#include <random>

#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter_params.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

/// @brief Fits a plane to a point cloud using variations of
/// the Random Sample Consensus (RANSAC) algorithm.
class RansacPlaneFitterCpu {
 public:
  /// @brief Constructs a RansacPlaneFitterCpu.
  RansacPlaneFitterCpu();

  /// @brief Fits a plane to a given point cloud.
  /// @param point_cloud The input vector of 3D points to which the plane is to
  /// be fitted.
  /// @return The best-fitting plane if found, otherwise std::nullopt.
  std::optional<Plane> fit(const std::vector<Vector3f>& point_cloud);

  /// A parameter setter
  /// @param ransac_distance_threshold the maximum distance a point
  /// can be from a plane to be considered an inlier.
  void ransac_distance_threshold_m(float distance_threshold);

  /// A parameter getter
  /// @returns distance_threshold the maximum distance a point
  /// can be from a plane to be considered an inlier.
  float ransac_distance_threshold_m() const;

  /// A parameter setter
  /// @param num_ransac_iterations to run the RANSAC algorithm.
  void num_ransac_iterations(int num_ransac_iterations);

  /// A parameter getter
  /// @returns the number of iterations to run the RANSAC algorithm.
  int num_ransac_iterations() const;

  /// A parameter setter
  /// @param ransac_type to use to calculate the inlier cost.
  void ransac_type(RansacType ransac_type);

  /// A parameter getter
  /// @returns ransac_type to use to calculate the inlier cost.
  RansacType ransac_type() const;

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  /// @brief Return the negative count of the number of inliers for a
  /// given plane and a point cloud.
  /// @param point_cloud The input vector of 3D points to check against the
  /// plane.
  /// @param plane The plane model used to determine inliers.
  /// @return The number of inliers within the specified distance threshold.
  int computePlaneToPointCloudInlierCostRansac(
      const std::vector<Vector3f>& point_cloud, const Plane& plane);

  /// @brief Return the redescending M-estimator cost for a given plane and a
  /// point cloud.
  /// @param point_cloud The input vector of 3D points to check against the
  /// plane.
  /// @param plane The plane model used to determine inliers.
  /// @return The number of inliers within the specified distance threshold.
  float computePlaneToPointCloudCostMSAC(
      const std::vector<Vector3f>& point_cloud, const Plane& plane);

  /// @brief Returns the cost of the specified RansacType.
  /// @param point_cloud The input vector of 3D points to check against the
  /// plane.
  /// @param plane The plane model used to determine inliers.
  /// @return The number of inliers within the specified distance threshold.
  float computePlaneToPointCloudCost(const std::vector<Vector3f>& point_cloud,
                                     const Plane& plane);

  // Random number generator to select points from the point cloud.
  std::random_device random_device_;
  std::mt19937 random_number_generator_;

  // Internal RANSAC configurations.
  // The maximum distance a point can be from a plane to be
  // considered an inlier.
  float ransac_distance_threshold_m_{
      kRansacDistanceThresholdMDesc.default_value};
  /// The number of iterations to run the RANSAC algorithm.
  int num_ransac_iterations_{kNumRansacIterationsDesc.default_value};
  // The ransac type to use to calculate the inlier cost.
  RansacType ransac_type_{kRansacTypeDesc.default_value};
};

}  // namespace nvblox