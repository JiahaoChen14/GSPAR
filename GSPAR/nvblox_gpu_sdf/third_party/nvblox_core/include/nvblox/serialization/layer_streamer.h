/*
Copyright 2023 NVIDIA CORPORATION

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

#include "nvblox/core/hash.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/serialization/internal/layer_streamer_traits.h"
#include "nvblox/utils/rates.h"

namespace nvblox {

/// A structure containing the specification of which blocks should be excluded
/// from layer streaming.
struct BlockExclusionParams {
  /// The center used for radius-based exclusion/
  std::optional<Vector3f> exclusion_center_m = std::nullopt;
  /// The height (in meters) above which blocks will be excluded from streaming.
  std::optional<float> exclusion_height_m = std::nullopt;
  /// The radius (in meters) used for radius-based block exclusion. Blocks
  /// further away from the center by this amount will not be streamed.
  std::optional<float> exclusion_radius_m = std::nullopt;
  /// The block size used during block exclusion.
  std::optional<float> block_size_m = std::nullopt;
};

// A function which tell the class whether to exclude a block from
// streaming
using ExcludeBlockFunctor = std::function<bool(const Index3D&)>;

// This map stores when the layer block was last published.
using BlockIndexToLastPublishedIndexMap = Index3DHashMapType<int64_t>::type;

/// Interface class which provides a non-templated base for all streamers
class LayerStreamerInterface {
 public:
  virtual ~LayerStreamerInterface() = default;
};

/// @brief Helps manage bandwidth used during layer streaming.
///
/// Note that this class is an abstract base class. Child classes have to
/// implement the compute priority function, which determines *which* blocks are
/// streamed when the number of potential blocks exceeds the requested limit.
template <class _LayerType>
class LayerStreamerBase : public LayerStreamerInterface {
 public:
  using LayerType = _LayerType;

  /// @brief Constructor
  LayerStreamerBase() = default;
  virtual ~LayerStreamerBase() = default;

  /// @brief Marks these block indices as candidates for streaming (usually
  /// because they have been touched by the reconstruction process).
  /// @param block_indices The indices of the candidate blocks
  void markIndicesCandidates(const std::vector<Index3D>& block_indices);

  /// @brief Returns N highest-priority blocks.
  ///
  /// This method modifies the list of tracked indices, removing those which are
  /// returned for streaming.
  /// @param num_block The number of blocks you want.
  /// @return The list of highest priority block indices.
  std::vector<Index3D> getNBlocks(const int num_blocks);

  /// @brief Returns highest priority blocks up to N bytes in size
  ///
  /// This method modifies the list of tracked indices, removing those which are
  /// returned for streaming.
  /// @param num_bytes The maximum number of bytes returned
  /// @param layer The layer which will be streamed. This is used to
  /// check the size (in bytes) of the blocks, so it should be updated
  /// before calling this function.
  /// @return The list of highest priority block indices.
  std::vector<Index3D> getNBytesOfBlocks(const size_t num_bytes,
                                         const LayerType& layer);

  /// @brief Returns highest priority serialized blocks up to N bytes
  ///
  /// @param num_bytes The maximum number of bytes returned
  /// @param layer layer to serialize
  /// @param cuda_stream Cuda stream.
  /// @return Serialized containing highest priority blocks
  std::shared_ptr<const SerializedLayerType<LayerType>>
  getNBytesOfSerializedBlocks(const size_t num_bytes, const LayerType& layer,
                              const CudaStream& cuda_stream);

  /// @brief Serialize all requested blocks, disregarding bandwidth and
  /// exclusion.
  ///
  /// @param layer layer to serialize
  /// @param block_indices block indices to serialize
  /// @param cuda_stream Cuda stream for GPU work
  std::shared_ptr<const SerializedLayerType<LayerType>> serializeAllBlocks(
      const LayerType& layer, const std::vector<Index3D>& block_indices,
      const CudaStream& cuda_stream);

  /// @brief Get the serialized result from the last computation
  std::shared_ptr<const SerializedLayerType<LayerType>> getSerializedLayer();

  /// @brief Returns the number of indices that are currently candidates
  /// for streaming.
  /// @return The number of indices we're keeping track of.
  int numCandidates() const;

  /// @brief Resets the streamer. Clearing all tracked indices.
  void clear();

  // Sets the exclusion functions being used.
  void setExclusionFunctors(
      std::vector<ExcludeBlockFunctor> exclude_block_functors);

 protected:
  // The function which determines a block's priority to be streamed.
  virtual std::vector<float> computePriorities(
      const std::vector<Index3D>& block_indices) const = 0;

  // Modifies the list of blocks eligible to be streamed in order to remove
  // some blocks before priorities are computed. Internally this function
  // calls the exclusion functors on each block.
  void excludeBlocks(std::vector<Index3D>* block_indices) const;

  // This struct which indicates whether a block should be streamed
  struct StreamStatus {
    // Should the query block be streamed
    bool should_block_be_streamed = false;
    // Is the query block invalid (and therefore we should stop tracking it)
    bool block_index_invalid = false;
    // Have we reached the streaming limit (and therefore should be stop
    // streaming)
    bool streaming_limit_reached = false;
  };
  // A function that determines if a block should be streamed.
  using StreamStatusFunctor = std::function<StreamStatus(const Index3D&)>;

  // A list of functors for testing blocks to be excluded from streaming
  // altogether.
  std::vector<ExcludeBlockFunctor> exclude_block_functors_;

  /// @brief Common function used by getNBlocks() and getNBytesBlocks().
  /// @param get_stream_status A functor which is called on candidate blocks
  /// and returns the status of the streaming process; for example, if the
  /// stream has reached its bandwidth capacity.
  /// @return The list of blocks to stream.
  std::vector<Index3D> getHighestPriorityBlocks(
      StreamStatusFunctor get_stream_status);

  // This set tracks the blocks which are candidates for streaming but have
  // not yet been streamed.
  Index3DSet index_set_;

  // Handles serialization of the layer
  SerializerType<LayerType> serializer_;
};

/// @brief A concrete child class of LayerStreamerBase.
///
/// This class implements a computePriorities() function which prioritizes
/// streaming the oldest Blocks, i.e. the Blocks that have not been
/// streamed to the longest time. Additionally, the streamer adds options for
/// excluding blocks above a certain height and blocks outside a certain
/// radius.
template <class _LayerType>
class LayerStreamerOldestBlocks : public LayerStreamerBase<_LayerType> {
 public:
  using LayerType = _LayerType;

  /// @brief Constructor
  LayerStreamerOldestBlocks() = default;
  virtual ~LayerStreamerOldestBlocks() = default;

  /// @brief Returns N oldest blocks for publishing
  /// @param num_blocks The number of blocks you want.
  /// @param exclusion_center_m Center point for radial exclusion
  /// @return The list of oldest block indices.
  std::vector<Index3D> getNBlocks(
      const int num_blocks, const BlockExclusionParams& block_exclusion_params);

  /// @brief Return N bytes of oldest blocks for publishing
  /// @param num_bytes The number of bytes of blocks to stream
  /// @param exclusion_center_m Center point for radial exclusion
  /// @return The list of oldest block indices.
  std::vector<Index3D> getNBytesOfBlocks(
      const size_t num_bytes, const LayerType& layer,
      const BlockExclusionParams& block_exclusion_params);

  /// @brief Returns highest priority serialized blocks up to N bytes
  ///
  /// @param num_bytes The maximum number of bytes returned
  /// @param layer layer to serialize
  /// @param exclusion_center_m Center point for radial exclusion
  /// @param cuda_stream Cuda stream.
  /// @return Serialized containing highest priority blocks
  std::shared_ptr<const SerializedLayerType<LayerType>>
  getNBytesOfSerializedBlocks(
      const size_t num_bytes, const LayerType& layer,
      const BlockExclusionParams& block_exclusion_params,
      const CudaStream& cuda_stream);

  /// @brief Serialize blocks. The amount of blocks to serialize is determined
  /// by limiting the data rate.
  ///
  /// @param layer Layer to serialize
  /// @param blocks_to_serialize blocks to serialize
  /// @param layer_nane Name of layer. Used for creating a timing::Rates
  /// @param maybe_exclusion_center Optional radial exclusion center
  /// @param bandwidth_limit_mbps Largest accepted bandwidth
  /// @param cuda_stream Cuda stream
  /// @return The serialized layer
  std::shared_ptr<const SerializedLayerType<LayerType>>
  estimateBandwidthAndSerialize(
      const LayerType& layer, const std::vector<Index3D>& blocks_to_serialize,
      const std::string& layer_name,
      const BlockExclusionParams& block_exclusion_params,
      const int bandwidth_limit_mbps, const CudaStream& cuda_stream);

 protected:
  // The method that defines this type of streamer. We compute the priority of
  // blocks as the oldest blocks as being the highest priority.
  virtual std::vector<float> computePriorities(
      const std::vector<Index3D>& block_indices) const override;
  float computePriority(const Index3D& block_index) const;

  // Called before blocks are returned to requester. Marks blocks as having
  // been streamed.
  void updateBlocksLastPublishIndex(const std::vector<Index3D>& block_indices);

  // Sets up the exclusion functors for this child class.
  // For this child class we (optionally) set up "blocks above height" and
  // "blocks outside radius" exclusion functors.
  void setupExclusionFunctors(
      const BlockExclusionParams& block_exclusion_params);

  static ExcludeBlockFunctor getExcludeAboveHeightFunctor(
      const float exclusion_height_m, const float block_size_m);

  static ExcludeBlockFunctor getExcludeOutsideRadiusFunctor(
      const float exclude_blocks_radius_m, const Vector3f& center_m,
      const float block_size_m);

  // The counts up with each call to getNBlocks(). It is used to indicate
  // the "when" blocks are returned for streaming.
  int64_t publishing_index_ = 0;

  BlockIndexToLastPublishedIndexMap last_published_map_;
};

constexpr float kLayerStreamerUnlimitedBandwidth = -1.0F;

using MeshLayerStreamerOldestBlocks = LayerStreamerOldestBlocks<MeshLayer>;
using TsdfLayerStreamerOldestBlocks = LayerStreamerOldestBlocks<TsdfLayer>;
using EsdfLayerStreamerOldestBlocks = LayerStreamerOldestBlocks<EsdfLayer>;
using OccupancyLayerStreamerOldestBlocks =
    LayerStreamerOldestBlocks<OccupancyLayer>;
using FreespaceLayerStreamerOldestBlocks =
    LayerStreamerOldestBlocks<FreespaceLayer>;
using ColorLayerStreamerOldestBlocks = LayerStreamerOldestBlocks<ColorLayer>;

}  // namespace nvblox
#include "nvblox/serialization/internal/impl/layer_streamer_impl.h"
