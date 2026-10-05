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

#include <memory>
#include <optional>

#include "nvblox/core/variadic_template_tools.h"

namespace nvblox {

template <typename LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerCakeStreamer::estimateBandwidthAndSerialize(
    const LayerType& layer, const std::vector<Index3D>& blocks_to_serialize,
    const std::string& layer_name,
    const BlockExclusionParams& block_exclusion_params,
    const int bandwidth_limit_mbps, const CudaStream& cuda_stream) {
  auto streamer_ptr = getPtr<LayerType>();
  if (streamer_ptr != nullptr) {
    return streamer_ptr->estimateBandwidthAndSerialize(
        layer, blocks_to_serialize, layer_name, block_exclusion_params,
        bandwidth_limit_mbps, cuda_stream);
  } else {
    LOG(WARNING) << "Called estimateBandwidthAndSerialize() for LayerType not "
                    "in the LayerCake. Doing nothing.";
    return std::shared_ptr<const SerializedLayerType<LayerType>>();
  }
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerCakeStreamer::serializeAllBlocks(const LayerType& layer,
                                      const std::vector<Index3D>& block_indices,
                                      const CudaStream& cuda_stream) {
  auto streamer_ptr = getPtr<LayerType>();
  if (streamer_ptr != nullptr) {
    return streamer_ptr->serializeAllBlocks(layer, block_indices, cuda_stream);
  } else {
    LOG(WARNING) << "Called serializeAllBlocks() for LayerType not "
                    "in the LayerCake. Doing nothing.";
    return std::shared_ptr<const SerializedLayerType<LayerType>>();
  }
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerCakeStreamer::getSerializedLayer() {
  auto streamer_ptr = getPtr<LayerType>();
  if (streamer_ptr != nullptr) {
    return streamer_ptr->getSerializedLayer();
  } else {
    LOG(WARNING) << "Called getSerializedLayer() for LayerType not "
                    "in the LayerCake. Doing nothing.";
    return std::shared_ptr<const SerializedLayerType<LayerType>>();
  }
}

template <typename LayerType>
void LayerCakeStreamer::add() {
  // Only one streamer per layer type
  if (streamers_.count(typeid(LayerType)) == 0) {
    streamers_.emplace(
        std::type_index(typeid(LayerType)),
        std::make_unique<LayerStreamerOldestBlocks<LayerType>>());
    LOG(INFO) << "Adding Streamer with type: "
              << typeid(LayerStreamerOldestBlocks<LayerType>).name()
              << " to LayerCake.";
  } else {
    LOG(WARNING) << "Request to add a Streamer that's already in the cake. "
                    "Currently we only support one of each type.";
  }
}

template <typename LayerType>
const LayerStreamerOldestBlocks<LayerType>* LayerCakeStreamer::getConstPtr()
    const {
  auto it = streamers_.find(std::type_index(typeid(LayerType)));
  if (it != streamers_.end()) {
    LayerStreamerInterface* base_ptr = it->second.get();
    LayerStreamerOldestBlocks<LayerType>* ptr =
        dynamic_cast<LayerStreamerOldestBlocks<LayerType>*>(base_ptr);
    CHECK_NOTNULL(ptr);
    return ptr;
  } else {
    LOG(WARNING) << "Request for a LayerType: " << typeid(LayerType).name()
                 << " which is not in the cake.";
    return nullptr;
  }
}

template <typename LayerType>
LayerStreamerOldestBlocks<LayerType>* LayerCakeStreamer::getPtr() {
  auto it = streamers_.find(std::type_index(typeid(LayerType)));
  if (it != streamers_.end()) {
    LayerStreamerInterface* base_ptr = it->second.get();
    LayerStreamerOldestBlocks<LayerType>* ptr =
        dynamic_cast<LayerStreamerOldestBlocks<LayerType>*>(base_ptr);
    CHECK_NOTNULL(ptr);
    return ptr;
  } else {
    LOG(WARNING) << "Request for a LayerType: " << typeid(LayerType).name()
                 << " which is not in the cake.";
    return nullptr;
  }
}

template <typename LayerType>
const LayerStreamerOldestBlocks<LayerType>& LayerCakeStreamer::get() const {
  const auto* ptr = getConstPtr<LayerType>();
  CHECK_NOTNULL(ptr);
  return *ptr;
}

template <typename... LayerTypes>
LayerCakeStreamer LayerCakeStreamer::create() {
  static_assert(
      unique_types<LayerTypes...>::value,
      "At the moment we only support LayerCakeStreamers containing unique "
      "LayerTypes.");
  LayerCakeStreamer cake_streamer;
  (cake_streamer.add<LayerTypes>(), ...);
  return cake_streamer;
}

}  // namespace nvblox
