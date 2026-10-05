/*
Copyright 2022-2024 NVIDIA CORPORATION

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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/integrators/internal/decay_integrator_base_params.h"
#include "nvblox/integrators/viewpoint.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// An options struct for specifying blocks excluded from decay.
struct DecayBlockExclusionOptions {
  /// A vector of block indices which should not be decayed in this decay step.
  std::vector<Index3D> block_indices_to_exclude = {};

  /// The center of radius-based block exclusion.
  std::optional<Vector3f> exclusion_center = std::nullopt;

  /// The radius of radius-based block exclusion.
  std::optional<float> exclusion_radius_m = std::nullopt;
};

/// A base class for the various decay integrators. It is specialized for
/// different voxel/layer types.
template <typename LayerType>
class DecayIntegratorBase {
 public:
  DecayIntegratorBase() = default;
  virtual ~DecayIntegratorBase() = default;

  DecayIntegratorBase(const DecayIntegratorBase&) = delete;
  DecayIntegratorBase& operator=(const DecayIntegratorBase&) const = delete;
  DecayIntegratorBase(DecayIntegratorBase&&) = delete;
  DecayIntegratorBase& operator=(const DecayIntegratorBase&&) const = delete;

  /// Decay all blocks. Fully decayed blocks (weight close to zero) will be
  /// deallocated if deallocate_decayed_blocks is true.
  ///
  /// @param layer_ptr    Layer to decay
  /// @param cuda_stream  Cuda stream for GPU work
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(LayerType* layer_ptr,
                                     const CudaStream& cuda_stream) = 0;
  /// Decay blocks. Blocks to decay can be excluded based on block index and/or
  /// distance to point.
  ///
  /// @param layer_ptr                 Layer to decay
  /// @param block_exclusion_options   Blocks to be excluded from decay
  /// @param cuda_stream               Cuda stream for GPU work
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      LayerType* layer_ptr,
      const DecayBlockExclusionOptions& block_exclusion_options,
      const CudaStream& cuda_stream) = 0;

  /// Decay blocks. Voxels can be excluded based on being in view.
  /// @param layer_ptr              Layer to decay
  /// @param view_exclusion_options Specifies view in which to exclude voxels
  /// @param cuda_stream            Cuda stream for GPU work.
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      LayerType* layer_ptr,
      const ViewBasedInclusionData& view_exclusion_options,
      const CudaStream& cuda_stream) = 0;

  /// Decay blocks. Optional block and voxel view exclusion.
  /// @param layer_ptr               Layer to decay
  /// @param block_exclusion_options Specifies blocks to be excluded from decay
  /// @param view_exclusion_options  Specifies view in which to exclude voxels
  /// @param cuda_stream             Cuda stream for GPU work.
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      LayerType* layer_ptr,
      const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
      const std::optional<ViewBasedInclusionData>& view_exclusion_options,
      const CudaStream& cuda_stream) = 0;

  /// A parameter getter
  /// The flag that controls if fully decayed block should be deallocated or
  /// not.
  /// @returns the deallocate_decayed_blocks flag
  bool deallocate_decayed_blocks() const;

  /// A parameter setter
  /// See deallocate_decayed_blocks().
  /// @param deallocate_decayed_blocks the new flag.
  void deallocate_decayed_blocks(bool deallocate_decayed_blocks);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  // Parameter for the decay step
  bool deallocate_decayed_blocks_{
      kDecayIntegratorDeallocateDecayedBlocks.default_value};
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/decay_integrator_impl.h"
