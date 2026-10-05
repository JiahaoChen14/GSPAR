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
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// Types of shapes that can be stored in ShapesUnion.
enum class ShapeType { kSphere, kAABB };

inline std::ostream& operator<<(std::ostream& os, const ShapeType& shape_type) {
  switch (shape_type) {
    case ShapeType::kSphere:
      os << "kSphere";
      break;
    case ShapeType::kAABB:
      os << "kAABB";
      break;
    default:
      break;
  }
  return os;
}

/// Union of different shapes.
union ShapesUnion {
  BoundingSphere sphere_;
  AxisAlignedBoundingBox aabb_;
  // NOTE: C++ automaticaly deletes default constructor and destructor of unions
  // containing non-trivial types. Adding them back here.
  ShapesUnion() {}
  ~ShapesUnion() {}
};

/// Class containing a bounding shape.
/// This could be a sphere or a bounding box.
class BoundingShape {
 public:
  /// Default constructor.
  BoundingShape();
  /// Constructor for sphere.
  BoundingShape(BoundingSphere sphere);
  /// Constructor for bounding box.
  BoundingShape(AxisAlignedBoundingBox aabb);
  /// Copy constructor.
  BoundingShape(const nvblox::BoundingShape& value);

  /// Getters for type and shapes.
  ShapeType type() const { return type_; }
  BoundingSphere sphere() const { return union_.sphere_; }
  AxisAlignedBoundingBox aabb() const { return union_.aabb_; }

  /// @brief Checking whether a point is inside the bounding shape.
  /// @param point The 3d point.
  /// @return Whether the points is inside the bounding shape.
  __host__ __device__ inline bool contains(const Vector3f& point) const;

  /// @brief Checking whether the bounding shape touches the block.
  /// @param block_index The index of the block.
  /// @param block_size The metric size of the block.
  /// @return Whether the bounding shape touches the block.
  bool touchesBlock(const Index3D& block_index, const float block_size) const;

 private:
  /// The type of the shape.
  ShapeType type_;
  /// The shape (stored as a union).
  ShapesUnion union_;
};

bool BoundingShape::contains(const Vector3f& point) const {
  switch (type_) {
    case ShapeType::kSphere:
      return union_.sphere_.contains(point);
    case ShapeType::kAABB:
      return union_.aabb_.contains(point);
    default:
      NVBLOX_ABORT("ShapeType not implemented");
      return false;
  }
}

}  // namespace nvblox
