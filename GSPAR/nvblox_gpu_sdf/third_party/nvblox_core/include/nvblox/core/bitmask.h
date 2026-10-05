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
#include <cstdint>

// Base class for wrapping an enum type into a bitmask.
// The enum is expected to contain incrementally increasing values, i.e. they do
// not have to be powers of 2.
// @tparam EnumType: integer-based enum type that is wrapped
template <typename EnumType>
class BitMask {
 public:
  using MaskType = int64_t;

  BitMask() = default;
  ~BitMask() = default;

  /// Construct from another bitmask
  BitMask(const BitMask& other);

  /// Construct from a raw mask
  BitMask(const EnumType& mask);

  /// Get the raw bitmask
  MaskType get() const;

  /// Assign from another bitmask
  BitMask& operator=(const EnumType& mask);

  /// Assign from a raw mask
  BitMask& operator=(const BitMask& other);

  // Logical OR between this and rhs. Used to set a bit in the mask.
  BitMask operator|(const BitMask& rhs) const;

  // Logical OR between this and rhs. Used to set a bit in the mask.
  BitMask operator|=(const BitMask& rhs);

  /// Boolean AND. Use to test if a given bit is set in the mask
  bool operator&(const BitMask& rhs) const;

  // Equality check
  bool operator==(const BitMask& other) const;

 private:
  // The bitmask
  MaskType bitmask_ = 0;
};

#include "nvblox/core/internal/impl/bitmask_impl.h"
