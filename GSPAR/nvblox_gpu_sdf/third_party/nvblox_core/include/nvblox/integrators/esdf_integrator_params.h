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

constexpr Param<float>::Description kEsdfIntegratorMaxDistanceMParamDesc{
    "esdf_integrator_max_distance_m", 2.f,
    "Maximum distance to compute the ESDF up to, in meters."};
constexpr Param<float>::Description kEsdfIntegratorMinWeightParamDesc{
    "esdf_integrator_min_weight", 1e-4,
    "Minimum weight of the TSDF to consider for inclusion in the ESDF."};
constexpr Param<float>::Description kEsdfIntegratorMaxSiteDistanceVoxParamDesc{
    "esdf_integrator_max_site_distance_vox", 1.f,
    "Maximum distance to consider a voxel within a surface for the ESDF "
    "calculation."};
constexpr Param<float>::Description kEsdfSliceMinHeightParamDesc{
    "esdf_slice_min_height", 0.f,
    "The minimum height, in meters, to consider obstacles part of the 2D "
    "ESDF slice."};
constexpr Param<float>::Description kEsdfSliceMaxHeightParamDesc{
    "esdf_slice_max_height", 1.f,
    "The maximum height, in meters, to consider obstacles part of the 2D "
    "ESDF slice."};
constexpr Param<float>::Description kEsdfSliceHeightParamDesc{
    "esdf_slice_height", 1.f,
    "The output slice height for the distance slice and ESDF pointcloud. "
    "Does not need to be within min and max height below. In units of "
    "meters."};
constexpr Param<float>::Description kSliceHeightAbovePlaneMParamDesc{
    "slice_height_above_plane_m", 0.0,
    "The height above the ground plane in meters at which we start "
    "slicing (from below)."};
constexpr Param<float>::Description kSliceHeightThicknessMParamDesc{
    "slice_height_thickness_m", 0.1,
    "The height of the slice (in meters) above the lower slice."};

struct EsdfIntegratorParams {
  Param<float> esdf_integrator_max_distance_m{
      kEsdfIntegratorMaxDistanceMParamDesc};
  Param<float> esdf_integrator_min_weight{kEsdfIntegratorMinWeightParamDesc};
  Param<float> esdf_integrator_max_site_distance_vox{
      kEsdfIntegratorMaxSiteDistanceVoxParamDesc};
  Param<float> esdf_slice_min_height{kEsdfSliceMinHeightParamDesc};
  Param<float> esdf_slice_max_height{kEsdfSliceMaxHeightParamDesc};
  Param<float> esdf_slice_height{kEsdfSliceHeightParamDesc};
  Param<float> slice_height_above_plane_m{kSliceHeightAbovePlaneMParamDesc};
  Param<float> slice_height_thickness_m{kSliceHeightThicknessMParamDesc};
};

}  // namespace nvblox
