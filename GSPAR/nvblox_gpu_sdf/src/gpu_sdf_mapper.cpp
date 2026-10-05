#include "nvblox_gpu_sdf/gpu_sdf_mapper.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include "nvblox/core/indexing.h"

namespace nvblox_gpu_sdf {

namespace {

nvblox::Transform toNvbloxTransform(const Transform& transform) {
  nvblox::Transform t = nvblox::Transform::Identity();
  Eigen::Matrix3f rotation;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) =
          transform.rotation[static_cast<std::size_t>(row * 3 + col)];
    }
  }
  t.linear() = rotation;
  t.translation() = Eigen::Vector3f(transform.translation[0],
                                    transform.translation[1],
                                    transform.translation[2]);
  return t;
}

nvblox::Camera toNvbloxCamera(const CameraIntrinsics& camera) {
  return nvblox::Camera(camera.fx, camera.fy, camera.cx, camera.cy,
                        camera.width, camera.height);
}

template <typename VoxelType, typename IsObserved, typename Value>
LayerStats layerStats(const nvblox::VoxelBlockLayer<VoxelType>& layer,
                      IsObserved is_observed, Value) {
  LayerStats stats;
  stats.blocks = layer.numAllocatedBlocks();
  for (const nvblox::Index3D& block_index : layer.getAllBlockIndices()) {
    const auto block = layer.getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    for (int x = 0; x < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++x) {
      for (int y = 0; y < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++y) {
        for (int z = 0; z < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide;
             ++z) {
          if (is_observed(block->voxels[x][y][z])) {
            ++stats.observed_voxels;
          }
        }
      }
    }
  }
  return stats;
}

template <typename VoxelType, typename IsObserved, typename Value>
bool writeObservedLayerToPly(const nvblox::VoxelBlockLayer<VoxelType>& layer,
                             const std::string& path, IsObserved is_observed,
                             Value value, LayerStats* stats_out) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  LayerStats stats;
  stats.blocks = layer.numAllocatedBlocks();
  for (const nvblox::Index3D& block_index : layer.getAllBlockIndices()) {
    const auto block = layer.getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    for (int x = 0; x < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++x) {
      for (int y = 0; y < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++y) {
        for (int z = 0; z < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide;
             ++z) {
          if (is_observed(block->voxels[x][y][z])) {
            ++stats.observed_voxels;
          }
        }
      }
    }
  }

  struct PlyVertex {
    float x;
    float y;
    float z;
    float intensity;
  };

  std::ofstream ply(path, std::ios::binary);
  if (!ply) {
    return false;
  }
  ply << "ply\n";
  ply << "format binary_little_endian 1.0\n";
  ply << "element vertex " << stats.observed_voxels << "\n";
  ply << "property float x\n";
  ply << "property float y\n";
  ply << "property float z\n";
  ply << "property float intensity\n";
  ply << "end_header\n";

  const float block_size = layer.block_size();
  std::vector<PlyVertex> batch;
  batch.reserve(1U << 20U);
  auto flushBatch = [&]() {
    if (batch.empty()) {
      return;
    }
    ply.write(reinterpret_cast<const char*>(batch.data()),
              static_cast<std::streamsize>(batch.size() * sizeof(PlyVertex)));
    batch.clear();
  };

  for (const nvblox::Index3D& block_index : layer.getAllBlockIndices()) {
    const auto block = layer.getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    for (int x = 0; x < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++x) {
      for (int y = 0; y < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide; ++y) {
        for (int z = 0; z < nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide;
             ++z) {
          const VoxelType& voxel = block->voxels[x][y][z];
          if (!is_observed(voxel)) {
            continue;
          }
          const nvblox::Vector3f p =
              nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
                  block_size, block_index, nvblox::Index3D(x, y, z));
          batch.push_back(PlyVertex{p.x(), p.y(), p.z(),
                                    value(voxel, layer.voxel_size())});
          if (batch.size() == batch.capacity()) {
            flushBatch();
          }
        }
      }
    }
  }
  flushBatch();
  if (!ply) {
    return false;
  }
  if (stats_out != nullptr) {
    *stats_out = stats;
  }
  return true;
}

bool writeNpyInt32Matrix(const std::filesystem::path& path,
                         const std::vector<std::int32_t>& data,
                         const std::size_t rows) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  std::ostringstream header_stream;
  header_stream << "{'descr': '<i4', 'fortran_order': False, 'shape': ("
                << rows << ", 3), }";
  std::string header = header_stream.str();
  const std::size_t preamble_size = 10U;
  const std::size_t newline_size = 1U;
  const std::size_t remainder =
      (preamble_size + header.size() + newline_size) % 16U;
  const std::size_t padding = remainder == 0U ? 0U : 16U - remainder;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
    return false;
  }
  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  const char version[] = {1, 0};
  const auto header_len = static_cast<std::uint16_t>(header.size());
  const char header_len_bytes[] = {
      static_cast<char>(header_len & 0xFFU),
      static_cast<char>((header_len >> 8U) & 0xFFU),
  };
  out.write(magic, sizeof(magic));
  out.write(version, sizeof(version));
  out.write(header_len_bytes, sizeof(header_len_bytes));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!data.empty()) {
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() *
                                           sizeof(std::int32_t)));
  }
  return static_cast<bool>(out);
}

std::string jsonArray3(const std::array<float, 3>& value) {
  std::ostringstream out;
  out << "[" << value[0] << ", " << value[1] << ", " << value[2] << "]";
  return out.str();
}

std::string jsonArray3(const std::array<int, 3>& value) {
  std::ostringstream out;
  out << "[" << value[0] << ", " << value[1] << ", " << value[2] << "]";
  return out.str();
}

}  // namespace

GpuSdfMapper::GpuSdfMapper(const MapperConfig& config)
    : config_(config),
      cuda_stream_(std::make_shared<nvblox::CudaStreamOwning>(
          cudaStreamNonBlocking)),
      tsdf_layer_(config.voxel_size_m, nvblox::MemoryType::kUnified),
      esdf_layer_(config.voxel_size_m, nvblox::MemoryType::kUnified),
      tsdf_integrator_(cuda_stream_),
      esdf_integrator_(cuda_stream_) {
  tsdf_integrator_.truncation_distance_vox(config_.truncation_distance_m /
                                           config_.voxel_size_m);
  tsdf_integrator_.max_weight(config_.max_weight);
  tsdf_integrator_.max_integration_distance_m(config_.max_depth_m);
  tsdf_integrator_.view_calculator().raycast_subsampling_factor(
      static_cast<unsigned int>(
          std::max(1, config_.raycast_subsampling_factor)));

  esdf_integrator_.max_esdf_distance_m(config_.max_esdf_distance_m);
  esdf_integrator_.max_site_distance_vox(config_.site_distance_vox);
  esdf_integrator_.min_weight(config_.min_tsdf_weight);
}

void GpuSdfMapper::integrateDepthFrame(
    const DepthImage& depth_image,
    const CameraIntrinsics& camera,
    const Transform& t_w_c) {
  nvblox::DepthImage nvblox_depth(nvblox::MemoryType::kUnified);
  nvblox_depth.copyFromAsync(depth_image.height, depth_image.width,
                             depth_image.depths_m.data(), *cuda_stream_);

  std::vector<nvblox::Index3D> updated_blocks;
  tsdf_integrator_.integrateFrame(nvblox_depth, toNvbloxTransform(t_w_c),
                                  toNvbloxCamera(camera), &tsdf_layer_,
                                  &updated_blocks);
  cuda_stream_->synchronize();
}

void GpuSdfMapper::updateEsdf() {
  const std::vector<nvblox::Index3D> blocks_to_update =
      tsdf_layer_.getAllBlockIndices();
  esdf_integrator_.integrateBlocks(tsdf_layer_, blocks_to_update,
                                   &esdf_layer_);
  cuda_stream_->synchronize();
}

LayerStats GpuSdfMapper::tsdfStats() const {
  cuda_stream_->synchronize();
  return layerStats(
      tsdf_layer_,
      [](const nvblox::TsdfVoxel& voxel) { return voxel.weight > 1.0e-4f; },
      [](const nvblox::TsdfVoxel& voxel, float) { return voxel.distance; });
}

LayerStats GpuSdfMapper::esdfStats() const {
  cuda_stream_->synchronize();
  return layerStats(
      esdf_layer_,
      [](const nvblox::EsdfVoxel& voxel) { return voxel.observed; },
      [](const nvblox::EsdfVoxel& voxel, float voxel_size) {
        float distance = voxel_size * std::sqrt(voxel.squared_distance_vox);
        return voxel.is_inside ? -distance : distance;
      });
}

bool GpuSdfMapper::writeTsdfObservedPly(const std::string& path,
                                        LayerStats* stats) const {
  cuda_stream_->synchronize();
  return writeObservedLayerToPly(
      tsdf_layer_,
      path,
      [](const nvblox::TsdfVoxel& voxel) { return voxel.weight > 1.0e-4f; },
      [](const nvblox::TsdfVoxel& voxel, float) { return voxel.distance; },
      stats);
}

bool GpuSdfMapper::writeEsdfObservedPly(const std::string& path,
                                        LayerStats* stats) const {
  cuda_stream_->synchronize();
  return writeObservedLayerToPly(
      esdf_layer_,
      path,
      [](const nvblox::EsdfVoxel& voxel) { return voxel.observed; },
      [](const nvblox::EsdfVoxel& voxel, float voxel_size) {
        float distance = voxel_size * std::sqrt(voxel.squared_distance_vox);
        return voxel.is_inside ? -distance : distance;
      },
      stats);
}

bool GpuSdfMapper::writeTsdfOccupancyGrid(
    const std::string& output_prefix, const float occupied_threshold_m,
    OccupancyGridStats* stats_out) const {
  cuda_stream_->synchronize();
  const std::filesystem::path prefix(output_prefix);
  if (!prefix.parent_path().empty()) {
    std::filesystem::create_directories(prefix.parent_path());
  }
  const std::filesystem::path npy_path =
      std::filesystem::path(prefix.string() + "_occupied_idx.npy");
  const std::filesystem::path json_path =
      std::filesystem::path(prefix.string() + ".json");

  constexpr int kSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;
  OccupancyGridStats stats;
  std::array<int, 3> min_global = {
      std::numeric_limits<int>::max(),
      std::numeric_limits<int>::max(),
      std::numeric_limits<int>::max(),
  };
  std::array<int, 3> max_global = {
      std::numeric_limits<int>::lowest(),
      std::numeric_limits<int>::lowest(),
      std::numeric_limits<int>::lowest(),
  };

  for (const nvblox::Index3D& block_index : tsdf_layer_.getAllBlockIndices()) {
    const auto block = tsdf_layer_.getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    for (int x = 0; x < kSide; ++x) {
      for (int y = 0; y < kSide; ++y) {
        for (int z = 0; z < kSide; ++z) {
          const nvblox::TsdfVoxel& voxel = block->voxels[x][y][z];
          if (voxel.weight <= 1.0e-4f) {
            continue;
          }
          ++stats.observed_voxels;
          const std::array<int, 3> global = {
              block_index.x() * kSide + x,
              block_index.y() * kSide + y,
              block_index.z() * kSide + z,
          };
          for (int axis = 0; axis < 3; ++axis) {
            min_global[axis] = std::min(min_global[axis], global[axis]);
            max_global[axis] = std::max(max_global[axis], global[axis]);
          }
          if (voxel.distance > occupied_threshold_m) {
            continue;
          }
          ++stats.occupied_voxels;
        }
      }
    }
  }

  if (stats.observed_voxels == 0U) {
    min_global = {0, 0, 0};
    max_global = {-1, -1, -1};
  }
  stats.origin_voxel_index = min_global;
  for (int axis = 0; axis < 3; ++axis) {
    stats.voxel_dims[axis] =
        std::max(0, max_global[axis] - min_global[axis] + 1);
    stats.origin_m[axis] =
        static_cast<float>(min_global[axis]) * tsdf_layer_.voxel_size();
    stats.scene_max_m[axis] =
        stats.origin_m[axis] +
        static_cast<float>(stats.voxel_dims[axis]) * tsdf_layer_.voxel_size();
  }

  std::vector<std::int32_t> occupied_idx;
  occupied_idx.reserve(stats.occupied_voxels * 3U);
  if (stats.occupied_voxels > 0U) {
    for (const nvblox::Index3D& block_index :
         tsdf_layer_.getAllBlockIndices()) {
      const auto block = tsdf_layer_.getBlockAtIndex(block_index);
      if (!block) {
        continue;
      }
      for (int x = 0; x < kSide; ++x) {
        for (int y = 0; y < kSide; ++y) {
          for (int z = 0; z < kSide; ++z) {
            const nvblox::TsdfVoxel& voxel = block->voxels[x][y][z];
            if (voxel.weight <= 1.0e-4f ||
                voxel.distance > occupied_threshold_m) {
              continue;
            }
            occupied_idx.push_back(static_cast<std::int32_t>(
                block_index.x() * kSide + x - min_global[0]));
            occupied_idx.push_back(static_cast<std::int32_t>(
                block_index.y() * kSide + y - min_global[1]));
            occupied_idx.push_back(static_cast<std::int32_t>(
                block_index.z() * kSide + z - min_global[2]));
          }
        }
      }
    }
  }

  if (!writeNpyInt32Matrix(npy_path, occupied_idx, stats.occupied_voxels)) {
    return false;
  }

  std::ofstream json(json_path);
  if (!json) {
    return false;
  }
  json << std::fixed;
  json.precision(6);
  json << "{\n";
  json << "  \"format\": \"nerficg_nvblox_occupancy_grid_v1\",\n";
  json << "  \"voxel_size\": " << tsdf_layer_.voxel_size() << ",\n";
  json << "  \"occupied_threshold_m\": " << occupied_threshold_m << ",\n";
  json << "  \"origin\": " << jsonArray3(stats.origin_m) << ",\n";
  json << "  \"scene_min\": " << jsonArray3(stats.origin_m) << ",\n";
  json << "  \"scene_max\": " << jsonArray3(stats.scene_max_m) << ",\n";
  json << "  \"voxel_dims\": " << jsonArray3(stats.voxel_dims) << ",\n";
  json << "  \"origin_voxel_index\": "
       << jsonArray3(stats.origin_voxel_index) << ",\n";
  json << "  \"num_tsdf_observed_voxels\": " << stats.observed_voxels
       << ",\n";
  json << "  \"num_occupied_voxels\": " << stats.occupied_voxels << ",\n";
  json << "  \"occupied_idx_npy\": \"" << npy_path.filename().string()
       << "\"\n";
  json << "}\n";
  if (!json) {
    return false;
  }
  if (stats_out != nullptr) {
    *stats_out = stats;
  }
  return true;
}

}  // namespace nvblox_gpu_sdf
