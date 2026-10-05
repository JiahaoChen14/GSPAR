#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace nvblox_gpu_sdf {

struct DepthImage {
  int width = 0;
  int height = 0;
  std::vector<float> depths_m;
};

struct CameraIntrinsics {
  int width = 0;
  int height = 0;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

// Transform from camera coordinates to world/map coordinates.
struct Transform {
  std::array<float, 9> rotation = {1.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f};
  std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
};

struct FrameInfo {
  std::size_t image_id = 0U;
  std::string image_name;
  std::string depth_path;
  CameraIntrinsics camera;
  Transform t_w_c;
};

struct Dataset {
  std::vector<FrameInfo> frames;
  float depth_scale_m = 1.0f;
};

struct ColmapDatasetOptions {
  std::string sparse_path;
  std::string depth_dir;
  bool prefer_geometric_depth = true;
};

class DatasetReader {
 public:
  static Dataset loadColmapDataset(const ColmapDatasetOptions& options);
  static DepthImage loadDepthImage(const std::string& depth_path,
                                   float depth_scale_m = 1.0f);
};

CameraIntrinsics scaledCameraForDepth(const CameraIntrinsics& camera,
                                      const DepthImage& depth_image);

}  // namespace nvblox_gpu_sdf
