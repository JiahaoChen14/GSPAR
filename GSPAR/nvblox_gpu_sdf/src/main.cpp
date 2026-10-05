#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "nvblox_gpu_sdf/dataset_io.h"
#include "nvblox_gpu_sdf/gpu_sdf_mapper.h"

namespace {

namespace fs = std::filesystem;

struct CliOptions {
  std::string colmap_sparse_path;
  std::string depth_dir;
  std::string output_dir = "output_nvblox_gpu";
  nvblox_gpu_sdf::MapperConfig mapper;
  bool max_depth_explicit = false;
  bool write_tsdf_ply = true;
  bool write_esdf_ply = true;
  bool write_occupancy_grid = true;
  float occupancy_threshold_m = 0.0f;
  std::size_t max_frames = 0U;
};

struct AdaptiveMaxDepthStats {
  bool enabled = false;
  bool fallback_used = false;
  std::string fallback_reason;
  float percentile = 90.0f;
  int sample_stride = 24;
  float minimum_max_depth_m = 20.0f;
  float round_up_multiple_m = 5.0f;
  std::size_t frames_sampled = 0U;
  std::size_t sampled_pixels = 0U;
  std::size_t valid_depth_samples = 0U;
  float valid_sample_fraction = 0.0f;
  float percentile_depth_m = 0.0f;
  float selected_max_depth_m = 20.0f;
};

struct PipelineTiming {
  double load_dataset_sec = 0.0;
  double adaptive_max_depth_sec = 0.0;
  double mapper_init_sec = 0.0;
  double depth_image_loading_sec = 0.0;
  double camera_preprocessing_sec = 0.0;
  double tsdf_integration_sec = 0.0;
  double esdf_update_sec = 0.0;
  double write_occupancy_grid_sec = 0.0;
  double write_tsdf_ply_sec = 0.0;
  double write_esdf_ply_sec = 0.0;
  double write_summary_sec = 0.0;
  double total_sec = 0.0;

  // Camera poses are supplied by a precomputed COLMAP sparse model. This
  // executable does not run SLAM, so its in-process SLAM time is always zero.
  double slamSec() const { return 0.0; }
  double tsdfSec() const { return tsdf_integration_sec; }
  double esdfSec() const { return esdf_update_sec; }
  double mappingSec() const { return tsdfSec() + esdfSec(); }
  double loadingSec() const {
    return load_dataset_sec + depth_image_loading_sec;
  }
  double calculationSec() const {
    return adaptive_max_depth_sec + mapper_init_sec +
           camera_preprocessing_sec + mappingSec();
  }
  double savingSec() const {
    return write_occupancy_grid_sec + write_tsdf_ply_sec +
           write_esdf_ply_sec + write_summary_sec;
  }
  double categorizedTotalSec() const {
    return loadingSec() + calculationSec() + savingSec();
  }
};

void printUsage(const char* executable) {
  std::cout
      << "Usage:\n"
      << "  " << executable
      << " --colmap_sparse /path/to/sparse --depth_dir /path/to/depths [options]\n"
      << "Options:\n"
      << "  [--output_dir OUTPUT_DIR]\n"
      << "  [--voxel_size 0.05] [--truncation_distance 0.20]\n"
      << "  [--max_weight 20] [--max_depth DEPTH] [--max_esdf_distance 5.0]\n"
      << "  [--site_distance_vox 1.0] [--min_tsdf_weight 1e-4]\n"
      << "  [--raycast_subsampling_factor 4] [--max_frames N]\n"
      << "  [--skip_tsdf_ply] [--skip_esdf_ply] [--skip_occupancy_grid]\n"
      << "  [--occupancy_threshold 0.0]\n"
      << "\n"
      << "If --max_depth is omitted, it is estimated as "
      << "max(20, ceil_to_5(valid depth p90)).\n";
}

CliOptions parseArgs(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto requireValue = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return argv[++i];
    };

    if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }
    if (argument == "--colmap_sparse") {
      options.colmap_sparse_path = requireValue(argument);
    } else if (argument == "--depth_dir") {
      options.depth_dir = requireValue(argument);
    } else if (argument == "--output_dir") {
      options.output_dir = requireValue(argument);
    } else if (argument == "--voxel_size") {
      options.mapper.voxel_size_m = std::stof(requireValue(argument));
    } else if (argument == "--truncation_distance") {
      options.mapper.truncation_distance_m = std::stof(requireValue(argument));
    } else if (argument == "--max_weight") {
      options.mapper.max_weight = std::stof(requireValue(argument));
    } else if (argument == "--max_depth") {
      options.mapper.max_depth_m = std::stof(requireValue(argument));
      options.max_depth_explicit = true;
    } else if (argument == "--max_esdf_distance") {
      options.mapper.max_esdf_distance_m = std::stof(requireValue(argument));
    } else if (argument == "--site_distance_vox") {
      options.mapper.site_distance_vox = std::stof(requireValue(argument));
    } else if (argument == "--min_tsdf_weight") {
      options.mapper.min_tsdf_weight = std::stof(requireValue(argument));
    } else if (argument == "--raycast_subsampling_factor") {
      options.mapper.raycast_subsampling_factor =
          std::stoi(requireValue(argument));
    } else if (argument == "--max_frames") {
      options.max_frames =
          static_cast<std::size_t>(std::stoull(requireValue(argument)));
    } else if (argument == "--skip_tsdf_ply") {
      options.write_tsdf_ply = false;
    } else if (argument == "--skip_esdf_ply") {
      options.write_esdf_ply = false;
    } else if (argument == "--skip_occupancy_grid") {
      options.write_occupancy_grid = false;
    } else if (argument == "--occupancy_threshold") {
      options.occupancy_threshold_m = std::stof(requireValue(argument));
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }

  if (options.colmap_sparse_path.empty() || options.depth_dir.empty()) {
    throw std::runtime_error("--colmap_sparse and --depth_dir are both required.");
  }
  return options;
}

double secondsSince(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

float roundUpToMultiple(const float value, const float multiple) {
  if (multiple <= 0.0f) {
    return value;
  }
  return std::ceil(value / multiple) * multiple;
}

float percentileValue(std::vector<float>* values, const float percentile) {
  if (values == nullptr || values->empty()) {
    throw std::runtime_error("Cannot compute percentile of empty values.");
  }
  const float clamped_percentile = std::clamp(percentile, 0.0f, 100.0f);
  const float rank =
      (clamped_percentile / 100.0f) * static_cast<float>(values->size());
  std::size_t index = 0U;
  if (rank > 0.0f) {
    index = static_cast<std::size_t>(std::ceil(rank)) - 1U;
  }
  index = std::min(index, values->size() - 1U);
  std::nth_element(values->begin(), values->begin() + index, values->end());
  return (*values)[index];
}

void readColmapDepthHeader(std::ifstream* file, const std::string& depth_path,
                           int* width, int* height, int* channels) {
  std::array<std::string, 3> header_fields;
  int delimiter_count = 0;
  char character = '\0';
  while (file->get(character)) {
    if (character == '&') {
      ++delimiter_count;
      if (delimiter_count == 3) {
        break;
      }
      continue;
    }
    if (delimiter_count >= 0 && delimiter_count < 3) {
      header_fields[static_cast<std::size_t>(delimiter_count)].push_back(
          character);
    }
  }
  if (delimiter_count != 3) {
    throw std::runtime_error("Malformed COLMAP depth map header: " +
                             depth_path);
  }

  *width = std::stoi(header_fields[0]);
  *height = std::stoi(header_fields[1]);
  *channels = std::stoi(header_fields[2]);
  if (*width <= 0 || *height <= 0 || *channels <= 0) {
    throw std::runtime_error("Invalid COLMAP depth map dimensions: " +
                             depth_path);
  }
}

void sampleDepthFile(const std::string& depth_path, const float depth_scale_m,
                     const int sample_stride,
                     std::vector<float>* valid_depth_samples,
                     std::size_t* sampled_pixels) {
  std::string extension = fs::path(depth_path).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](const unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (extension == ".png") {
    const nvblox_gpu_sdf::DepthImage image =
        nvblox_gpu_sdf::DatasetReader::loadDepthImage(depth_path,
                                                      depth_scale_m);
    const int stride = std::max(1, sample_stride);
    for (int row_idx = 0; row_idx < image.height; row_idx += stride) {
      for (int col_idx = 0; col_idx < image.width; col_idx += stride) {
        ++(*sampled_pixels);
        const float depth = image.depths_m[
            static_cast<std::size_t>(row_idx) *
                static_cast<std::size_t>(image.width) +
            static_cast<std::size_t>(col_idx)];
        if (std::isfinite(depth) && depth > 0.0f) {
          valid_depth_samples->push_back(depth);
        }
      }
    }
    return;
  }

  std::ifstream file(depth_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP depth map: " + depth_path);
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  readColmapDepthHeader(&file, depth_path, &width, &height, &channels);

  const int stride = std::max(1, sample_stride);
  const std::size_t row_value_count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  const std::streamoff row_bytes =
      static_cast<std::streamoff>(row_value_count * sizeof(float));
  std::vector<float> row(row_value_count);

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    if (row_idx % stride != 0) {
      file.seekg(row_bytes, std::ios::cur);
      if (!file) {
        throw std::runtime_error("COLMAP depth map data is truncated: " +
                                 depth_path);
      }
      continue;
    }

    file.read(reinterpret_cast<char*>(row.data()),
              static_cast<std::streamsize>(row_bytes));
    if (!file) {
      throw std::runtime_error("COLMAP depth map data is truncated: " +
                               depth_path);
    }
    for (int col_idx = 0; col_idx < width; col_idx += stride) {
      ++(*sampled_pixels);
      float depth =
          row[static_cast<std::size_t>(col_idx) *
              static_cast<std::size_t>(channels)] *
          depth_scale_m;
      if (std::isfinite(depth) && depth > 0.0f) {
        valid_depth_samples->push_back(depth);
      }
    }
  }
}

AdaptiveMaxDepthStats estimateAdaptiveMaxDepth(
    const nvblox_gpu_sdf::Dataset& dataset) {
  AdaptiveMaxDepthStats stats;
  stats.enabled = true;
  std::vector<float> valid_depth_samples;

  for (const nvblox_gpu_sdf::FrameInfo& frame : dataset.frames) {
    sampleDepthFile(frame.depth_path, dataset.depth_scale_m,
                    stats.sample_stride, &valid_depth_samples,
                    &stats.sampled_pixels);
    ++stats.frames_sampled;
  }

  stats.valid_depth_samples = valid_depth_samples.size();
  if (stats.sampled_pixels > 0U) {
    stats.valid_sample_fraction =
        static_cast<float>(stats.valid_depth_samples) /
        static_cast<float>(stats.sampled_pixels);
  }

  if (valid_depth_samples.empty()) {
    stats.fallback_used = true;
    stats.fallback_reason = "no_valid_depth_samples";
    stats.selected_max_depth_m = stats.minimum_max_depth_m;
    return stats;
  }

  stats.percentile_depth_m =
      percentileValue(&valid_depth_samples, stats.percentile);
  stats.selected_max_depth_m =
      std::max(stats.minimum_max_depth_m,
               roundUpToMultiple(stats.percentile_depth_m,
                                 stats.round_up_multiple_m));
  return stats;
}

void writeTimingJson(const fs::path& output_dir, const CliOptions& options,
                     const AdaptiveMaxDepthStats& adaptive,
                     const PipelineTiming& timing) {
  std::ofstream json(output_dir / "timing_stats.json");
  json << std::fixed << std::setprecision(6);
  json << "{\n";
  json << "  \"max_depth\": {\n";
  json << "    \"mode\": \""
       << (options.max_depth_explicit ? "explicit" : "adaptive") << "\",\n";
  json << "    \"max_depth_m\": " << options.mapper.max_depth_m << ",\n";
  json << "    \"adaptive_enabled\": "
       << (adaptive.enabled ? "true" : "false") << ",\n";
  json << "    \"fallback_used\": "
       << (adaptive.fallback_used ? "true" : "false") << ",\n";
  json << "    \"fallback_reason\": \"" << adaptive.fallback_reason
       << "\",\n";
  json << "    \"percentile\": " << adaptive.percentile << ",\n";
  json << "    \"sample_stride\": " << adaptive.sample_stride << ",\n";
  json << "    \"minimum_max_depth_m\": " << adaptive.minimum_max_depth_m
       << ",\n";
  json << "    \"round_up_multiple_m\": " << adaptive.round_up_multiple_m
       << ",\n";
  json << "    \"frames_sampled\": " << adaptive.frames_sampled << ",\n";
  json << "    \"sampled_pixels\": " << adaptive.sampled_pixels << ",\n";
  json << "    \"valid_depth_samples\": " << adaptive.valid_depth_samples
       << ",\n";
  json << "    \"valid_sample_fraction\": "
       << adaptive.valid_sample_fraction << ",\n";
  json << "    \"percentile_depth_m\": " << adaptive.percentile_depth_m
       << ",\n";
  json << "    \"selected_max_depth_m\": "
       << adaptive.selected_max_depth_m << "\n";
  json << "  },\n";
  json << "  \"slam\": {\n";
  json << "    \"executed_in_process\": false,\n";
  json << "    \"pose_source\": \"precomputed_colmap_sparse\",\n";
  json << "    \"note\": \"SLAM is external; only its saved camera poses are loaded\"\n";
  json << "  },\n";
  json << "  \"timing_sec\": {\n";
  json << "    \"loading\": " << timing.loadingSec() << ",\n";
  json << "    \"calculation\": " << timing.calculationSec() << ",\n";
  json << "    \"slam\": " << timing.slamSec() << ",\n";
  json << "    \"tsdf\": " << timing.tsdfSec() << ",\n";
  json << "    \"esdf\": " << timing.esdfSec() << ",\n";
  json << "    \"tsdf_esdf\": " << timing.mappingSec() << ",\n";
  json << "    \"saving\": " << timing.savingSec() << ",\n";
  json << "    \"categorized_total\": " << timing.categorizedTotalSec()
       << ",\n";
  json << "    \"wall_total\": " << timing.total_sec << ",\n";
  json << "    \"stages\": {\n";
  json << "      \"load_colmap_and_index_depth\": "
       << timing.load_dataset_sec << ",\n";
  json << "      \"adaptive_max_depth\": "
       << timing.adaptive_max_depth_sec << ",\n";
  json << "      \"mapper_initialization\": " << timing.mapper_init_sec
       << ",\n";
  json << "      \"depth_image_loading\": "
       << timing.depth_image_loading_sec << ",\n";
  json << "      \"camera_preprocessing\": "
       << timing.camera_preprocessing_sec << ",\n";
  json << "      \"slam\": " << timing.slamSec() << ",\n";
  json << "      \"tsdf_integration\": " << timing.tsdfSec() << ",\n";
  json << "      \"esdf_update\": " << timing.esdfSec() << ",\n";
  json << "      \"write_occupancy_grid\": "
       << timing.write_occupancy_grid_sec << ",\n";
  json << "      \"write_tsdf_ply\": " << timing.write_tsdf_ply_sec
       << ",\n";
  json << "      \"write_esdf_ply\": " << timing.write_esdf_ply_sec
       << ",\n";
  json << "      \"write_summary\": " << timing.write_summary_sec << "\n";
  json << "    }\n";
  json << "  }\n";
  json << "}\n";
}

void writeSummary(const fs::path& output_dir, const CliOptions& options,
                  const nvblox_gpu_sdf::Dataset& dataset,
                  const nvblox_gpu_sdf::LayerStats& tsdf_stats,
                  const nvblox_gpu_sdf::LayerStats& esdf_stats,
                  const AdaptiveMaxDepthStats& adaptive,
                  const PipelineTiming& timing) {
  std::ofstream summary(output_dir / "summary.txt");
  summary << std::fixed << std::setprecision(6);
  summary << "backend nvblox_gpu_sdf\n";
  summary << "gpu_accelerated true\n";
  summary << "slam_executed_in_process false\n";
  summary << "slam_pose_source precomputed_colmap_sparse\n";
  summary << "frames_integrated " << dataset.frames.size() << "\n";
  summary << "voxel_size_m " << options.mapper.voxel_size_m << "\n";
  summary << "truncation_distance_m "
          << options.mapper.truncation_distance_m << "\n";
  summary << "max_weight " << options.mapper.max_weight << "\n";
  summary << "max_depth_m " << options.mapper.max_depth_m << "\n";
  summary << "max_depth_mode "
          << (options.max_depth_explicit ? "explicit" : "adaptive") << "\n";
  if (adaptive.enabled) {
    summary << "adaptive_max_depth_percentile " << adaptive.percentile << "\n";
    summary << "adaptive_max_depth_sample_stride "
            << adaptive.sample_stride << "\n";
    summary << "adaptive_max_depth_minimum_m "
            << adaptive.minimum_max_depth_m << "\n";
    summary << "adaptive_max_depth_round_up_multiple_m "
            << adaptive.round_up_multiple_m << "\n";
    summary << "adaptive_max_depth_frames_sampled "
            << adaptive.frames_sampled << "\n";
    summary << "adaptive_max_depth_sampled_pixels "
            << adaptive.sampled_pixels << "\n";
    summary << "adaptive_max_depth_valid_samples "
            << adaptive.valid_depth_samples << "\n";
    summary << "adaptive_max_depth_valid_sample_fraction "
            << adaptive.valid_sample_fraction << "\n";
    summary << "adaptive_max_depth_p90_m "
            << adaptive.percentile_depth_m << "\n";
    summary << "adaptive_max_depth_selected_m "
            << adaptive.selected_max_depth_m << "\n";
    summary << "adaptive_max_depth_fallback_used "
            << (adaptive.fallback_used ? "true" : "false") << "\n";
    if (adaptive.fallback_used) {
      summary << "adaptive_max_depth_fallback_reason "
              << adaptive.fallback_reason << "\n";
    }
  }
  summary << "max_esdf_distance_m " << options.mapper.max_esdf_distance_m
          << "\n";
  summary << "site_distance_vox " << options.mapper.site_distance_vox << "\n";
  summary << "min_tsdf_weight " << options.mapper.min_tsdf_weight << "\n";
  summary << "raycast_subsampling_factor "
          << options.mapper.raycast_subsampling_factor << "\n";
  summary << "max_frames " << options.max_frames << "\n";
  if (!dataset.frames.empty()) {
    summary << "first_camera_width " << dataset.frames.front().camera.width
            << "\n";
    summary << "first_camera_height " << dataset.frames.front().camera.height
            << "\n";
  }
  summary << "tsdf_blocks " << tsdf_stats.blocks << "\n";
  summary << "tsdf_observed_voxels " << tsdf_stats.observed_voxels << "\n";
  summary << "esdf_blocks " << esdf_stats.blocks << "\n";
  summary << "esdf_observed_voxels " << esdf_stats.observed_voxels << "\n";
  summary << "write_tsdf_ply " << (options.write_tsdf_ply ? "true" : "false")
          << "\n";
  summary << "write_esdf_ply " << (options.write_esdf_ply ? "true" : "false")
          << "\n";
  summary << "write_occupancy_grid "
          << (options.write_occupancy_grid ? "true" : "false") << "\n";
  summary << "occupancy_threshold_m " << options.occupancy_threshold_m
          << "\n";
  summary << "timing_loading_sec " << timing.loadingSec() << "\n";
  summary << "timing_calculation_sec " << timing.calculationSec() << "\n";
  summary << "timing_slam_sec " << timing.slamSec() << "\n";
  summary << "timing_tsdf_sec " << timing.tsdfSec() << "\n";
  summary << "timing_esdf_sec " << timing.esdfSec() << "\n";
  summary << "timing_tsdf_esdf_sec " << timing.mappingSec() << "\n";
  summary << "timing_saving_sec " << timing.savingSec() << "\n";
  summary << "timing_categorized_total_sec "
          << timing.categorizedTotalSec() << "\n";
  summary << "timing_wall_total_sec " << timing.total_sec << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  try {
    const auto total_start = std::chrono::steady_clock::now();
    CliOptions options = parseArgs(argc, argv);
    PipelineTiming timing;
    AdaptiveMaxDepthStats adaptive_max_depth;

    auto phase_start = std::chrono::steady_clock::now();
    nvblox_gpu_sdf::Dataset dataset =
        nvblox_gpu_sdf::DatasetReader::loadColmapDataset(
            nvblox_gpu_sdf::ColmapDatasetOptions{options.colmap_sparse_path,
                                                 options.depth_dir});
    if (options.max_frames > 0U && options.max_frames < dataset.frames.size()) {
      dataset.frames.resize(options.max_frames);
    }
    timing.load_dataset_sec = secondsSince(phase_start);

    if (!options.max_depth_explicit) {
      phase_start = std::chrono::steady_clock::now();
      adaptive_max_depth = estimateAdaptiveMaxDepth(dataset);
      options.mapper.max_depth_m = adaptive_max_depth.selected_max_depth_m;
      timing.adaptive_max_depth_sec = secondsSince(phase_start);
    }

    phase_start = std::chrono::steady_clock::now();
    nvblox_gpu_sdf::GpuSdfMapper mapper(options.mapper);
    timing.mapper_init_sec = secondsSince(phase_start);

    for (std::size_t frame_idx = 0; frame_idx < dataset.frames.size();
         ++frame_idx) {
      const nvblox_gpu_sdf::FrameInfo& frame = dataset.frames[frame_idx];
      phase_start = std::chrono::steady_clock::now();
      const nvblox_gpu_sdf::DepthImage depth_image =
          nvblox_gpu_sdf::DatasetReader::loadDepthImage(frame.depth_path,
                                                        dataset.depth_scale_m);
      timing.depth_image_loading_sec += secondsSince(phase_start);

      phase_start = std::chrono::steady_clock::now();
      const nvblox_gpu_sdf::CameraIntrinsics camera =
          nvblox_gpu_sdf::scaledCameraForDepth(frame.camera, depth_image);
      timing.camera_preprocessing_sec += secondsSince(phase_start);

      phase_start = std::chrono::steady_clock::now();
      mapper.integrateDepthFrame(depth_image, camera, frame.t_w_c);
      timing.tsdf_integration_sec += secondsSince(phase_start);
    }

    phase_start = std::chrono::steady_clock::now();
    mapper.updateEsdf();
    timing.esdf_update_sec = secondsSince(phase_start);

    const fs::path output_dir = fs::path(options.output_dir);
    fs::create_directories(output_dir);
    nvblox_gpu_sdf::LayerStats tsdf_stats;
    nvblox_gpu_sdf::LayerStats esdf_stats;
    nvblox_gpu_sdf::OccupancyGridStats occupancy_stats;
    phase_start = std::chrono::steady_clock::now();
    if (options.write_occupancy_grid) {
      if (!mapper.writeTsdfOccupancyGrid(
              (output_dir / "tsdf_occupancy_grid").string(),
              options.occupancy_threshold_m, &occupancy_stats)) {
        throw std::runtime_error("Failed to write tsdf_occupancy_grid");
      }
      tsdf_stats.blocks = mapper.tsdfLayer().numAllocatedBlocks();
      tsdf_stats.observed_voxels = occupancy_stats.observed_voxels;
    } else {
      tsdf_stats = mapper.tsdfStats();
    }
    timing.write_occupancy_grid_sec = secondsSince(phase_start);

    phase_start = std::chrono::steady_clock::now();
    if (options.write_tsdf_ply) {
      if (!mapper.writeTsdfObservedPly(
              (output_dir / "tsdf_observed.ply").string(), &tsdf_stats)) {
        throw std::runtime_error("Failed to write tsdf_observed.ply");
      }
    }
    timing.write_tsdf_ply_sec = secondsSince(phase_start);

    phase_start = std::chrono::steady_clock::now();
    if (options.write_esdf_ply) {
      if (!mapper.writeEsdfObservedPly(
              (output_dir / "esdf_observed.ply").string(), &esdf_stats)) {
        throw std::runtime_error("Failed to write esdf_observed.ply");
      }
    } else {
      esdf_stats = mapper.esdfStats();
    }
    timing.write_esdf_ply_sec = secondsSince(phase_start);

    phase_start = std::chrono::steady_clock::now();
    timing.total_sec = secondsSince(total_start);
    writeSummary(output_dir, options, dataset, tsdf_stats, esdf_stats,
                 adaptive_max_depth, timing);
    writeTimingJson(output_dir, options, adaptive_max_depth, timing);
    timing.write_summary_sec = secondsSince(phase_start);
    timing.total_sec = secondsSince(total_start);
    writeSummary(output_dir, options, dataset, tsdf_stats, esdf_stats,
                 adaptive_max_depth, timing);
    writeTimingJson(output_dir, options, adaptive_max_depth, timing);

    std::cout << std::fixed << std::setprecision(3)
              << "[NVBLOX] done: frames=" << dataset.frames.size()
              << ", max_depth=" << options.mapper.max_depth_m << "m"
              << ", observed_voxels=" << tsdf_stats.observed_voxels;
    if (options.write_occupancy_grid) {
      std::cout << ", occupied_voxels=" << occupancy_stats.occupied_voxels;
    }
    std::cout << ", slam=" << timing.slamSec()
              << "s, tsdf=" << timing.tsdfSec()
              << "s, esdf=" << timing.esdfSec()
              << "s, tsdf_esdf=" << timing.mappingSec()
              << "s, calculation=" << timing.calculationSec()
              << "s, total=" << timing.total_sec << "s\n"
              << "[NVBLOX] output: " << output_dir.string() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
