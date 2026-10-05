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

namespace nvblox {

/// Takes a list of ProjectiveIntegrators and makes them share a ViewpointCache
/// @tparam ...Args Variadic template
/// @param ...integrator_ptr A list of pointers to projective integrators which
/// should share.
template <typename... Args>
void shareViewpointCaches(Args*... integrator_ptr);

}  // namespace nvblox

#include "nvblox/mapper/internal/impl/mapper_common_impl.h"
