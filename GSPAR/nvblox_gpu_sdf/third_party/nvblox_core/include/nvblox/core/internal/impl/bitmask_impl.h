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
#include <glog/logging.h>

#include <cstdlib>

template <typename EnumType>
BitMask<EnumType>::BitMask(const BitMask<EnumType>& other)
    : bitmask_(other.bitmask_) {}

template <typename EnumType>
BitMask<EnumType>::BitMask(const EnumType& enum_value)
    : bitmask_(1LL << static_cast<int>(enum_value)) {
  CHECK_LE(static_cast<size_t>(enum_value), 8 * sizeof(MaskType));
}

template <typename EnumType>
BitMask<EnumType>& BitMask<EnumType>::operator=(const EnumType& enum_value) {
  CHECK_LE(enum_value, 8 * sizeof(MaskType));
  bitmask_ = (1LL << static_cast<MaskType>(enum_value));
  return *this;
}

template <typename EnumType>
BitMask<EnumType>& BitMask<EnumType>::operator=(
    const BitMask<EnumType>& other) {
  bitmask_ = other.bitmask_;
  return *this;
}

template <typename EnumType>
BitMask<EnumType> BitMask<EnumType>::operator|(
    const BitMask<EnumType>& rhs) const {
  BitMask<EnumType> ret;
  ret.bitmask_ = bitmask_ | rhs.bitmask_;
  return ret;
}

template <typename EnumType>
BitMask<EnumType> BitMask<EnumType>::operator|=(const BitMask<EnumType>& rhs) {
  *this = *this | rhs;
  return *this;
}

template <typename EnumType>
bool BitMask<EnumType>::operator&(const BitMask<EnumType>& rhs) const {
  return static_cast<bool>(static_cast<MaskType>(bitmask_) &
                           static_cast<MaskType>(rhs.bitmask_));
}

template <typename EnumType>
bool BitMask<EnumType>::operator==(const BitMask& other) const {
  return bitmask_ == other.bitmask_;
}

template <typename EnumType>
typename BitMask<EnumType>::MaskType BitMask<EnumType>::get() const {
  return bitmask_;
}
