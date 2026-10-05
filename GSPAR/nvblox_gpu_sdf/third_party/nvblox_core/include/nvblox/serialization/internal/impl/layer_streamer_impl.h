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
#include "nvblox/serialization/layer_streamer.h"

#include <numeric>

namespace nvblox {

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerStreamerBase<LayerType>::getSerializedLayer() {
  return serializer_.getSerializedLayer();
}

template <class LayerType>
int LayerStreamerBase<LayerType>::numCandidates() const {
  return index_set_.size();
}

template <class LayerType>
void LayerStreamerBase<LayerType>::clear() {
  return index_set_.clear();
}

template <class LayerType>
void LayerStreamerBase<LayerType>::markIndicesCandidates(
    const std::vector<Index3D>& block_indices) {
  index_set_.insert(block_indices.begin(), block_indices.end());
}

template <class LayerType>
void LayerStreamerBase<LayerType>::setExclusionFunctors(
    std::vector<ExcludeBlockFunctor> exclude_block_functors) {
  exclude_block_functors_ = exclude_block_functors;
};

template <class LayerType>
std::vector<Index3D> LayerStreamerBase<LayerType>::getNBlocks(
    const int num_blocks) {
  // Define a functor that counts the number of blocks streamed so far.
  int num_blocks_streamed = 0;
  StreamStatusFunctor stream_n_blocks_functor =
      [&num_blocks_streamed, num_blocks](const Index3D&) -> StreamStatus {
    // If we have enough bandwidth stream, otherwise finish streaming
    const bool should_block_be_streamed = num_blocks_streamed < num_blocks;
    if (should_block_be_streamed) {
      ++num_blocks_streamed;
    }
    return {.should_block_be_streamed = should_block_be_streamed,
            .block_index_invalid = false,
            .streaming_limit_reached = !should_block_be_streamed};
  };

  // Stream N highest priority blocks
  return getHighestPriorityBlocks(stream_n_blocks_functor);
}

template <class LayerType>
std::vector<Index3D> LayerStreamerBase<LayerType>::getNBytesOfBlocks(
    const size_t num_bytes, const LayerType& layer) {
  // Define a functor that counts the number of bytes streamed so far.
  size_t num_bytes_streamed = 0;
  StreamStatusFunctor stream_n_bytes_functor =
      [&num_bytes_streamed, num_bytes,
       &layer](const Index3D& idx) -> StreamStatus {
    StreamStatus status;
    const typename LayerType::BlockType::ConstPtr block_ptr =
        layer.getBlockAtIndex(idx);
    // If the mesh block has been deallocated, don't stream and indicate invalid
    if (!block_ptr) {
      return {.should_block_be_streamed = false,
              .block_index_invalid = true,
              .streaming_limit_reached = false};
    }
    // The bytes that would be sent if we sent this block
    num_bytes_streamed += sizeInBytes(block_ptr.get());
    // If we have enough bandwidth send, otherwise stop streaming.
    const bool should_stream = num_bytes_streamed < num_bytes;
    return {.should_block_be_streamed = should_stream,
            .block_index_invalid = false,
            .streaming_limit_reached = !should_stream};
  };

  // Stream N highest priority blocks
  return getHighestPriorityBlocks(stream_n_bytes_functor);
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerStreamerBase<LayerType>::getNBytesOfSerializedBlocks(
    const size_t num_bytes, const LayerType& layer,
    const CudaStream& cuda_stream) {
  return serializer_.serialize(layer, getNBytesOfMeshBlocks(num_bytes, layer),
                               cuda_stream);
}

template <class LayerType>
std::vector<Index3D> LayerStreamerBase<LayerType>::getHighestPriorityBlocks(
    StreamStatusFunctor get_stream_status) {
  // Convert the set of indices to a vector for index-based random access
  std::vector<Index3D> index_vec;
  std::copy(index_set_.begin(), index_set_.end(),
            std::back_inserter(index_vec));

  // Clear the set. We'll repopulate it with the un-streamed vertices later
  index_set_.clear();

  // Before we compute any priorities we can exclude blocks
  excludeBlocks(&index_vec);

  // Compute the priority of each block
  const std::vector<float> priorities = computePriorities(index_vec);

  // Sort the priorities
  std::vector<int> indices(priorities.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Sort high to low
  std::sort(indices.begin(), indices.end(),
            [&](const int a, const int b) -> bool {
              return priorities[a] > priorities[b];
            });

  // Split the index_vec into highest priority for streaming and the lower
  // priority to store in the class for next time.
  std::vector<Index3D> high_priority_blocks;
  int insert_remaining_start_idx = -1;
  for (int i = 0; i < static_cast<int>(index_vec.size()); i++) {
    // Get the index of the i'th high priority block.
    const int sorted_index = indices[i];
    const Index3D& block_index = index_vec[sorted_index];
    // Call out to the functor to see what we should do with this block.
    const StreamStatus stream_status = get_stream_status(block_index);
    // Stream
    if (stream_status.should_block_be_streamed) {
      high_priority_blocks.push_back(block_index);
    }
    // Stay - block should not be streamed but *is* valid.
    else if (!stream_status.block_index_invalid) {
      index_set_.insert(block_index);
    }
    // Streaming limit reached, stop testing
    if (stream_status.streaming_limit_reached) {
      insert_remaining_start_idx = i + 1;
      break;
    }
  }
  // If we terminated early, add the untested blocks back into the tracking set.
  if (insert_remaining_start_idx > 0) {
    for (size_t i = insert_remaining_start_idx; i < indices.size(); i++) {
      const int sorted_index = indices[i];
      const Index3D block_index = index_vec[sorted_index];
      index_set_.insert(block_index);
    }
  }
  // Sanity check that the set isn't getting too big.
  // NOTE(alexmillane): This number is approximately 50m*50m*10m at 0.05m
  // voxels. At this point you really should be using radius-based exclusion (in
  // the child class LayerStreamerOldestBlocks).
  constexpr size_t kNumBlocksWarningThreshold = 500000ul;
  if (index_set_.size() > kNumBlocksWarningThreshold) {
    LOG(WARNING) << "The number of tracked mesh blocks for streaming is "
                    "getting very large: "
                 << index_set_.size()
                 << ". Consider adding some form of block exclusion.";
  }
  return high_priority_blocks;
}

template <class LayerType>
void LayerStreamerBase<LayerType>::excludeBlocks(
    std::vector<Index3D>* block_indices) const {
  if (exclude_block_functors_.size() == 0) {
    // The base implementation is to not exclude any blocks.
    return;
  }
  // Exclude blocks not meeting the exclusion functions
  std::vector<Index3D> index_vec_not_excluded;
  index_vec_not_excluded.reserve(block_indices->size());
  for (const Index3D& block_idx : *block_indices) {
    bool exclude = false;
    for (const ExcludeBlockFunctor& exclude_block_functor :
         exclude_block_functors_) {
      if (exclude_block_functor(block_idx)) {
        // Exclusion function fired, move to the next block.
        exclude = true;
        break;
      };
    }
    // If none of the exclusion functions fire, we add this block
    if (!exclude) {
      index_vec_not_excluded.push_back(block_idx);
    }
  }
  // Rewrite the block indices
  *block_indices = std::move(index_vec_not_excluded);
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerStreamerBase<LayerType>::serializeAllBlocks(
    const LayerType& layer, const std::vector<Index3D>& block_indices,
    const CudaStream& cuda_stream) {
  return serializer_.serialize(layer, block_indices, cuda_stream);
}

template <class LayerType>
std::vector<Index3D> LayerStreamerOldestBlocks<LayerType>::getNBlocks(
    const int num_blocks, const BlockExclusionParams& block_exclusion_params) {
  setupExclusionFunctors(block_exclusion_params);
  const std::vector<Index3D> block_indices =
      LayerStreamerBase<LayerType>::getNBlocks(num_blocks);
  // Mark these blocks as streamed.
  updateBlocksLastPublishIndex(block_indices);
  return block_indices;
}

template <class LayerType>
std::vector<Index3D> LayerStreamerOldestBlocks<LayerType>::getNBytesOfBlocks(
    const size_t num_bytes, const LayerType& layer,
    const BlockExclusionParams& block_exclusion_params) {
  // Calls the base class method, after setting up the exclusion functors.
  setupExclusionFunctors(block_exclusion_params);
  const std::vector<Index3D> block_indices =
      LayerStreamerBase<LayerType>::getNBytesOfBlocks(num_bytes, layer);
  // Mark these blocks as streamed.
  updateBlocksLastPublishIndex(block_indices);
  return block_indices;
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerStreamerOldestBlocks<LayerType>::getNBytesOfSerializedBlocks(
    const size_t num_bytes, const LayerType& layer,
    const BlockExclusionParams& block_exclusion_params,
    const CudaStream& cuda_stream) {
  return LayerStreamerBase<LayerType>::serializer_.serialize(
      layer, getNBytesOfBlocks(num_bytes, layer, block_exclusion_params),
      cuda_stream);
}

template <class LayerType>
std::vector<float> LayerStreamerOldestBlocks<LayerType>::computePriorities(
    const std::vector<Index3D>& block_indices) const {
  std::vector<float> priorities;
  std::transform(
      block_indices.begin(), block_indices.end(),
      std::back_inserter(priorities),
      [&](const Index3D& block_index) { return computePriority(block_index); });
  return priorities;
}

template <class LayerType>
float LayerStreamerOldestBlocks<LayerType>::computePriority(
    const Index3D& block_index) const {
  // More recently published blocks should have lower priority, so we negate
  // the last published index (more recent blocks have a higher index value,
  // and therefore lower priority)
  const auto it = last_published_map_.find(block_index);
  if (it == last_published_map_.end()) {
    return static_cast<float>(std::numeric_limits<int64_t>::max());
  } else {
    return static_cast<float>(-1 * it->second);
  }
}

template <class LayerType>
void LayerStreamerOldestBlocks<LayerType>::updateBlocksLastPublishIndex(
    const std::vector<Index3D>& block_indices) {
  // Go through the blocks and mark them as having been streamed.
  for (const Index3D& block_idx : block_indices) {
    // Insert a new publishing index or update the old one.
    last_published_map_[block_idx] = publishing_index_;
  }
  ++publishing_index_;
  CHECK_LT(publishing_index_, std::numeric_limits<int64_t>::max());
}

template <class LayerType>
void LayerStreamerOldestBlocks<LayerType>::setupExclusionFunctors(
    const BlockExclusionParams& block_exclusion_params) {
  // If requested exclude blocks
  std::vector<ExcludeBlockFunctor> exclusion_functors;
  // Exclude based on height
  if (block_exclusion_params.exclusion_height_m.has_value() &&
      block_exclusion_params.block_size_m.has_value() &&
      block_exclusion_params.exclusion_height_m.value() > 0.0) {
    exclusion_functors.push_back(getExcludeAboveHeightFunctor(
        block_exclusion_params.exclusion_height_m.value(),
        block_exclusion_params.block_size_m.value()));
  }
  // Exclude based on radius
  if (block_exclusion_params.block_size_m.has_value() &&
      block_exclusion_params.exclusion_center_m.has_value() &&
      block_exclusion_params.exclusion_radius_m.has_value() &&
      block_exclusion_params.exclusion_radius_m.value() > 0.0) {
    exclusion_functors.push_back(getExcludeOutsideRadiusFunctor(
        block_exclusion_params.exclusion_radius_m.value(),
        block_exclusion_params.exclusion_center_m.value(),
        block_exclusion_params.block_size_m.value()));
  }
  LayerStreamerBase<LayerType>::setExclusionFunctors(exclusion_functors);
}

template <class LayerType>
ExcludeBlockFunctor
LayerStreamerOldestBlocks<LayerType>::getExcludeAboveHeightFunctor(
    const float exclusion_height_m, const float block_size_m) {
  // Create a functor which returns true if a blocks minimum height is above
  // a limit.
  return [exclusion_height_m, block_size_m](const Index3D& idx) -> bool {
    // Exclude block if it's low corner/plane is above the exclusion
    // limit
    const float low_z_m = static_cast<float>(idx.z()) * block_size_m;
    return low_z_m > exclusion_height_m;
  };
}

template <class LayerType>
ExcludeBlockFunctor
LayerStreamerOldestBlocks<LayerType>::getExcludeOutsideRadiusFunctor(
    const float exclude_blocks_radius_m, const Vector3f& center_m,
    const float block_size_m) {
  // Square the radius outside
  const float exclude_blocks_radius_squared_m2 =
      exclude_blocks_radius_m * exclude_blocks_radius_m;
  // Create a functor which returns true if the block center is a greater radius
  // from the passed center.
  return [exclude_blocks_radius_squared_m2, center_m,
          block_size_m](const Index3D& idx) -> bool {
    // Calculate the blocks center position
    const Vector3f block_center =
        getCenterPositionFromBlockIndex(block_size_m, idx);
    const float block_radius_squared_m2 =
        (block_center - center_m).squaredNorm();
    return block_radius_squared_m2 > exclude_blocks_radius_squared_m2;
  };
}

template <class LayerType>
std::shared_ptr<const SerializedLayerType<LayerType>>
LayerStreamerOldestBlocks<LayerType>::estimateBandwidthAndSerialize(
    const LayerType& layer, const std::vector<Index3D>& blocks_to_serialize,
    const std::string& layer_name,
    const BlockExclusionParams& block_exclusion_params,
    const int bandwidth_limit_mbps, const CudaStream& cuda_stream) {
  LayerStreamerBase<LayerType>::markIndicesCandidates(blocks_to_serialize);

  // Measure tick rate of requests to determine how many bytes we
  // should stream. clamp measurement to avoid instabilities when there are
  // few rate measurements.
  const std::string timer_name = "layer_streamer/" + layer_name;

  timing::Rates::tick(timer_name);
  constexpr float kMinRateHz = 1.F;
  constexpr float kMaxRateHz = 100.F;
  const float measured_rate_hz =
      std::max(kMinRateHz,
               std::min(kMaxRateHz, timing::Rates::getMeanRateHz(timer_name)));

  const float measured_update_period_s = 1.0f / measured_rate_hz;
  const float megabits_per_update =
      bandwidth_limit_mbps * measured_update_period_s;
  constexpr float kMegabitsToBytes = 1e6f / 8.0f;

  // If requested bandwidth is negative, we stream as much as we can
  const size_t num_bytes_to_stream =
      (bandwidth_limit_mbps < 0.F)
          ? std::numeric_limits<size_t>::max()
          : static_cast<size_t>(megabits_per_update * kMegabitsToBytes);

  auto ret = getNBytesOfSerializedBlocks(num_bytes_to_stream, layer,
                                         block_exclusion_params, cuda_stream);
  DLOG(INFO) << "Streamed " << ret->block_indices.size() << " " << layer_name
             << " blocks out of " << blocks_to_serialize.size()
             << ". num_bytes_to_stream: " << num_bytes_to_stream << std::endl;
  return ret;
}

};  // namespace nvblox
