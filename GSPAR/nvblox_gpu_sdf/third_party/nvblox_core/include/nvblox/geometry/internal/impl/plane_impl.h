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

#include <cassert>

namespace nvblox {

__host__ __device__ inline Plane::Plane() {
  d_ = 0.0;
  normal_ = Vector3f{1.0, 0.0, 0.0};
}

__host__ __device__ inline Plane::Plane(const Vector3f& normal, const float d) {
  d_ = (d);
  normal_ = normal.normalized();
}

__host__ __device__ inline Plane::Plane(const Vector3f& normal,
                                        const Vector3f& point)
    : Plane(normal.normalized(), -point.dot(normal.normalized())) {}

__host__ __device__ inline bool Plane::planeFromPoints(const Vector3f& p_a,
                                                       const Vector3f& p_b,
                                                       const Vector3f& p_c,
                                                       Plane* plane_out) {
  // All points are the same
  if (p_a == p_b || p_a == p_c || p_b == p_c) {
    return false;
  }
  const Vector3f ab = p_b - p_a;
  const Vector3f ac = p_c - p_a;
  const Vector3f ab_cross_ac = ab.cross(ac);

  // Points must not be colinear
  if (ab_cross_ac.isZero(1e-6)) {
    return false;
  }
  *plane_out = Plane(ab_cross_ac.normalized(), p_a);
  return true;
}

__host__ __device__ inline float Plane::signedDistance(
    const Vector3f& p) const {
  return normal_.dot(p) + d_;
}

__host__ __device__ inline Vector3f Plane::project(const Vector3f& p) const {
  return p - signedDistance(p) * normal_;
}

__host__ __device__ inline float Plane::getHeightAtXY(
    const Vector2f& xy) const {
  // To get the height on the plane we rearrange the plane equation:
  // \vec{n} \dot \vec{x} + d = 0
  // z = - (n_1*x + n_2*y + d) / n_3
  // Note that n_3 must not be zero, which would be a vertical plane.
  assert(normal_.z() > 1.0e-4);
  return -1.0f * (normal_.x() * xy.x() + normal_.y() * xy.y() + offset()) /
         normal_.z();
}

}  // namespace nvblox
