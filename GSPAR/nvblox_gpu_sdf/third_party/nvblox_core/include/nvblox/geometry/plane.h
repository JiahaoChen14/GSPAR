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

namespace nvblox {

/// Represents a plane.
class Plane {
 public:
  /// @brief Default constructor.
  /// @details Initializes a plane with a unit x normal vector and zero offset.
  __host__ __device__ inline Plane();

  /// Create a Plane from a normal and an offset.
  /// @param normal Normal vector.
  /// @param d The offset in the plane equation: n \dot p + d = 0
  __host__ __device__ inline Plane(const Vector3f& normal, const float d);

  /// Create a Plane from a normal and a point.
  /// @param normal Normal vector.
  /// @param point A point on the plane.
  __host__ __device__ inline Plane(const Vector3f& normal,
                                   const Vector3f& point);
  /// Create a Plane from three points.
  /// @param p_a A point on the plane.
  /// @param p_b A second point on the plane.
  /// @param p_c A third point on the plane.
  /// @param plane_out The output plane.
  /// @return If the points are collinear or not distinct, false is returned.
  __host__ __device__ static inline bool planeFromPoints(const Vector3f& p_a,
                                                         const Vector3f& p_b,
                                                         const Vector3f& p_c,
                                                         Plane* plane_out);

  /// The minimal signed distance from p to the plane.
  /// @param p A point.
  /// @return The signed minimal distance to the plane.
  __host__ __device__ inline float signedDistance(const Vector3f& p) const;

  /// Return the projection of a point p on the plane.
  /// @param p The point to be projected.
  /// @return The projected point.
  __host__ __device__ inline Vector3f project(const Vector3f& p) const;

  /// Return the projection of a point p on the plane.
  /// @param p The point to be projected.
  /// @return The projected point.
  /// @param xy The point in x,y.
  __host__ __device__ inline float getHeightAtXY(const Vector2f& xy) const;

  /// Getter.
  /// @return The normal vector.
  __host__ __device__ inline const Vector3f& normal() const { return normal_; }

  /// Getter.
  /// @return The offset.
  __host__ __device__ inline float d() const { return d_; }

  /// Getter.
  /// @return The offset.
  __host__ __device__ inline float offset() const { return d_; }

 protected:
  /// The plane normal unit-vector.
  Vector3f normal_;

  /// The constant that makes the plane equation true n \dot p + d = 0
  float d_;
};

}  // namespace nvblox

#include "nvblox/geometry/internal/impl/plane_impl.h"
