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

template <typename... Args>
void shareViewpointCaches(Args*... integrator_ptr) {
  // Make the camera integrators share the same viewpoint cache.
  std::shared_ptr<ViewpointCache> raycasting_viewpoint_cache =
      std::make_shared<ViewpointCache>();
  (integrator_ptr->view_calculator().set_viewpoint_cache(
       raycasting_viewpoint_cache,
       ViewCalculator::CalculationType::kRaycasting),
   ...);
  std::shared_ptr<ViewpointCache> planes_viewpoint_cache =
      std::make_shared<ViewpointCache>();
  (integrator_ptr->view_calculator().set_viewpoint_cache(
       planes_viewpoint_cache, ViewCalculator::CalculationType::kPlanes),
   ...);
}

}  // namespace nvblox
