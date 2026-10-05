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
#include <typeindex>

#include "nvblox/core/types.h"
#include "nvblox/serialization/layer_streamer.h"

namespace nvblox {

class LayerCakeStreamer {
 public:
  LayerCakeStreamer() = default;

  /// @brief Serialize blocks. The amount of blocks to serialize is determined
  /// by limiting the data rate.
  ///
  /// @param layer Layer to serialize
  /// @param blocks_to_serialize blocks to serialize
  /// @param layer_name Name of layer. Used for creating a timing::Rates
  /// @param block_exclusion_params Specifies which blocks should be excluded
  /// from streaming.
  /// @param bandwidth_limit_mbps Largest accepted bandwidth
  /// @param cuda_stream Cuda stream
  /// @return The serialized layer
  template <typename LayerType>
  std::shared_ptr<const SerializedLayerType<LayerType>>
  estimateBandwidthAndSerialize(
      const LayerType& layer, const std::vector<Index3D>& blocks_to_serialize,
      const std::string& layer_name,
      const BlockExclusionParams& block_exclusion_params,
      const int bandwidth_limit_mbps, const CudaStream& cuda_stream);

  /// @brief Serialize all requested blocks, disregarding bandwidth and
  /// exclusion.
  ///
  /// @param layer layer to serialize
  /// @param block_indices block indices to serialize
  /// @param cuda_stream Cuda stream for GPU work
  template <typename LayerType>
  std::shared_ptr<const SerializedLayerType<LayerType>> serializeAllBlocks(
      const LayerType& layer, const std::vector<Index3D>& block_indices,
      const CudaStream& cuda_stream);

  /// @brief Get the serialized result from the last computation
  template <typename LayerType>
  std::shared_ptr<const SerializedLayerType<LayerType>> getSerializedLayer();

  /// Adds a new LayerStreamer to the collection
  /// Note that only a single layer of each type may be added. Calls to try to
  /// add multiple layers of the same type will have no effect.
  /// @tparam LayerType The type layer to be serialized.
  template <typename LayerType>
  void add();

  /// Get a LayerStreamer by type.
  /// @tparam LayerType The type of the Layer streamed by the LayerStreamer.
  /// @return A pointer to the LayerStreamer, or nullptr if no layer of this
  /// type exists in the collection.
  template <typename LayerType>
  LayerStreamerOldestBlocks<LayerType>* getPtr();
  /// See \ref getPtr.
  template <typename LayerType>
  const LayerStreamerOldestBlocks<LayerType>* getConstPtr() const;

  /// Get a LayerStreamer by type.
  /// Terminates the program if the layer cake does not contain a layer of the
  /// specified type. See \ref getPtr for a more resilient interface.
  /// @tparam LayerType The type of the Layer streamed by the LayerStreamer.
  /// @return A reference to the LayerStreamer in the collection.
  template <typename LayerType>
  const LayerStreamerOldestBlocks<LayerType>& get() const;

  /// Factory. Creates a LayerCakeStreamer containing several LayersStreamers.
  /// @tparam ...LayerTypes A list of Layer types streamed to the
  /// LayerStreamers.
  /// @return A LayerCakeStreamer containing the layers.
  template <typename... LayerTypes>
  static LayerCakeStreamer create();

 private:
  // Holds a LayerStreamer per LayerType
  std::unordered_map<std::type_index, std::unique_ptr<LayerStreamerInterface>>
      streamers_;
};

}  // namespace nvblox
#include "nvblox/serialization/internal/impl/layer_cake_streamer_impl.h"
