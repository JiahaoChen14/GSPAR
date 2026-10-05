#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "nvblox_gpu_sdf/dataset_io.h"
#include "nvblox/core/cuda_stream.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox_gpu_sdf {

struct MapperConfig {
  float voxel_size_m = 0.05f;
  float truncation_distance_m = 0.20f;
  float max_weight = 20.0f;
  float max_depth_m = 20.0f;
  float max_esdf_distance_m = 5.0f;
  float site_distance_vox = 1.0f;
  float min_tsdf_weight = 1.0e-4f;
  int raycast_subsampling_factor = 4;
};

struct LayerStats {
  std::size_t blocks = 0U;
  std::size_t observed_voxels = 0U;
};

struct OccupancyGridStats {
  std::size_t observed_voxels = 0U;
  std::size_t occupied_voxels = 0U;
  std::array<int, 3> origin_voxel_index = {0, 0, 0};
  std::array<int, 3> voxel_dims = {0, 0, 0};
  std::array<float, 3> origin_m = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> scene_max_m = {0.0f, 0.0f, 0.0f};
};

class GpuSdfMapper {
 public:
  explicit GpuSdfMapper(const MapperConfig& config);

  void integrateDepthFrame(const DepthImage& depth_image,
                           const CameraIntrinsics& camera,
                           const Transform& t_w_c);
  void updateEsdf();

  LayerStats tsdfStats() const;
  LayerStats esdfStats() const;

  bool writeTsdfObservedPly(const std::string& path,
                            LayerStats* stats = nullptr) const;
  bool writeEsdfObservedPly(const std::string& path,
                            LayerStats* stats = nullptr) const;
  bool writeTsdfOccupancyGrid(const std::string& output_prefix,
                              float occupied_threshold_m = 0.0f,
                              OccupancyGridStats* stats = nullptr) const;

  const MapperConfig& config() const { return config_; }
  const nvblox::TsdfLayer& tsdfLayer() const { return tsdf_layer_; }
  const nvblox::EsdfLayer& esdfLayer() const { return esdf_layer_; }

 private:
  MapperConfig config_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  nvblox::TsdfLayer tsdf_layer_;
  nvblox::EsdfLayer esdf_layer_;
  nvblox::ProjectiveTsdfIntegrator tsdf_integrator_;
  nvblox::EsdfIntegrator esdf_integrator_;
};

}  // namespace nvblox_gpu_sdf
