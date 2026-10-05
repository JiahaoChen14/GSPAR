#include "nvblox_gpu_sdf/dataset_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "nvblox/io/internal/thirdparty/stb_image.h"

namespace nvblox_gpu_sdf {

namespace {

namespace fs = std::filesystem;

struct CameraRecord {
  std::uint32_t camera_id = 0U;
  CameraIntrinsics intrinsics;
};

struct ImageRecord {
  std::uint32_t image_id = 0U;
  std::uint32_t camera_id = 0U;
  std::string image_name;
  Transform t_w_c;
};

struct CameraModel {
  const char* name = "";
  int num_params = 0;
};

std::string trim(const std::string& input) {
  const std::size_t first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const std::size_t last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1U);
}

bool startsWithCommentOrEmpty(const std::string& line) {
  const std::string stripped = trim(line);
  return stripped.empty() || stripped.front() == '#';
}

template <typename T>
T readBinary(std::istream& stream) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("Unexpected end of binary COLMAP file.");
  }
  return value;
}

std::string readNullTerminatedString(std::istream& stream) {
  std::string value;
  char character = '\0';
  while (stream.get(character)) {
    if (character == '\0') {
      return value;
    }
    value.push_back(character);
  }
  throw std::runtime_error("Unexpected end of binary COLMAP string.");
}

CameraModel cameraModelFromId(const int model_id) {
  switch (model_id) {
    case 0:
      return {"SIMPLE_PINHOLE", 3};
    case 1:
      return {"PINHOLE", 4};
    case 2:
      return {"SIMPLE_RADIAL", 4};
    case 3:
      return {"RADIAL", 5};
    case 4:
      return {"OPENCV", 8};
    case 5:
      return {"OPENCV_FISHEYE", 8};
    case 6:
      return {"FULL_OPENCV", 12};
    case 7:
      return {"FOV", 5};
    case 8:
      return {"SIMPLE_RADIAL_FISHEYE", 4};
    case 9:
      return {"RADIAL_FISHEYE", 5};
    case 10:
      return {"THIN_PRISM_FISHEYE", 12};
    default:
      throw std::runtime_error("Unsupported COLMAP camera model id: " +
                               std::to_string(model_id));
  }
}

CameraIntrinsics intrinsicsFromParams(const std::string& model,
                                      const std::uint64_t width,
                                      const std::uint64_t height,
                                      const std::vector<double>& params) {
  CameraIntrinsics camera;
  camera.width = static_cast<int>(width);
  camera.height = static_cast<int>(height);

  if (model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" ||
      model == "RADIAL" || model == "SIMPLE_RADIAL_FISHEYE" ||
      model == "RADIAL_FISHEYE") {
    if (params.size() < 3U) {
      throw std::runtime_error("COLMAP camera model " + model +
                               " has too few parameters.");
    }
    camera.fx = static_cast<float>(params[0]);
    camera.fy = static_cast<float>(params[0]);
    camera.cx = static_cast<float>(params[1]);
    camera.cy = static_cast<float>(params[2]);
    return camera;
  }

  if (model == "PINHOLE" || model == "OPENCV" ||
      model == "OPENCV_FISHEYE" || model == "FULL_OPENCV" ||
      model == "FOV" || model == "THIN_PRISM_FISHEYE") {
    if (params.size() < 4U) {
      throw std::runtime_error("COLMAP camera model " + model +
                               " has too few parameters.");
    }
    camera.fx = static_cast<float>(params[0]);
    camera.fy = static_cast<float>(params[1]);
    camera.cx = static_cast<float>(params[2]);
    camera.cy = static_cast<float>(params[3]);
    return camera;
  }

  throw std::runtime_error("Unsupported COLMAP camera model: " + model);
}

std::array<double, 9> rotationFromColmapQuaternion(double qw, double qx,
                                                   double qy, double qz) {
  const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (norm <= 0.0) {
    throw std::runtime_error("Invalid zero-length COLMAP quaternion.");
  }
  qw /= norm;
  qx /= norm;
  qy /= norm;
  qz /= norm;

  return {1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
          2.0 * qx * qy - 2.0 * qz * qw,
          2.0 * qx * qz + 2.0 * qy * qw,
          2.0 * qx * qy + 2.0 * qz * qw,
          1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
          2.0 * qy * qz - 2.0 * qx * qw,
          2.0 * qx * qz - 2.0 * qy * qw,
          2.0 * qy * qz + 2.0 * qx * qw,
          1.0 - 2.0 * qx * qx - 2.0 * qy * qy};
}

Transform worldFromCameraTransform(double qw, double qx, double qy, double qz,
                                   double tx, double ty, double tz) {
  const std::array<double, 9> r_c_w =
      rotationFromColmapQuaternion(qw, qx, qy, qz);
  const std::array<double, 3> t_c_w = {tx, ty, tz};

  Transform t_w_c;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      t_w_c.rotation[static_cast<std::size_t>(row * 3 + col)] =
          static_cast<float>(r_c_w[static_cast<std::size_t>(col * 3 + row)]);
    }
  }
  for (int row = 0; row < 3; ++row) {
    double value = 0.0;
    for (int col = 0; col < 3; ++col) {
      value -= static_cast<double>(
                   t_w_c.rotation[static_cast<std::size_t>(row * 3 + col)]) *
               t_c_w[static_cast<std::size_t>(col)];
    }
    t_w_c.translation[static_cast<std::size_t>(row)] =
        static_cast<float>(value);
  }
  return t_w_c;
}

std::unordered_map<std::uint32_t, CameraRecord> readCamerasText(
    const fs::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP cameras text file: " +
                             path.string());
  }

  std::unordered_map<std::uint32_t, CameraRecord> cameras;
  std::string line;
  while (std::getline(file, line)) {
    if (startsWithCommentOrEmpty(line)) {
      continue;
    }
    std::istringstream stream(line);
    std::uint32_t camera_id = 0U;
    std::string model;
    std::uint64_t width = 0U;
    std::uint64_t height = 0U;
    stream >> camera_id >> model >> width >> height;
    if (!stream) {
      throw std::runtime_error("Malformed COLMAP cameras.txt line: " + line);
    }

    std::vector<double> params;
    double param = 0.0;
    while (stream >> param) {
      params.push_back(param);
    }
    cameras[camera_id] =
        CameraRecord{camera_id, intrinsicsFromParams(model, width, height,
                                                     params)};
  }
  return cameras;
}

std::vector<ImageRecord> readImagesText(const fs::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP images text file: " +
                             path.string());
  }

  std::vector<ImageRecord> images;
  std::string line;
  while (std::getline(file, line)) {
    if (startsWithCommentOrEmpty(line)) {
      continue;
    }

    std::istringstream stream(line);
    std::uint32_t image_id = 0U;
    double qw = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    std::uint32_t camera_id = 0U;
    stream >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >>
        camera_id;
    if (!stream) {
      throw std::runtime_error("Malformed COLMAP images.txt line: " + line);
    }

    std::string image_name;
    std::getline(stream, image_name);
    image_name = trim(image_name);
    if (image_name.empty()) {
      throw std::runtime_error("COLMAP image line has an empty image name.");
    }

    images.push_back(ImageRecord{
        image_id, camera_id, image_name,
        worldFromCameraTransform(qw, qx, qy, qz, tx, ty, tz)});

    std::getline(file, line);
  }
  return images;
}

std::unordered_map<std::uint32_t, CameraRecord> readCamerasBinary(
    const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP cameras binary file: " +
                             path.string());
  }

  const std::uint64_t num_cameras = readBinary<std::uint64_t>(file);
  std::unordered_map<std::uint32_t, CameraRecord> cameras;
  for (std::uint64_t idx = 0U; idx < num_cameras; ++idx) {
    const std::uint32_t camera_id = readBinary<std::uint32_t>(file);
    const int model_id = readBinary<int>(file);
    const std::uint64_t width = readBinary<std::uint64_t>(file);
    const std::uint64_t height = readBinary<std::uint64_t>(file);
    const CameraModel model = cameraModelFromId(model_id);

    std::vector<double> params(static_cast<std::size_t>(model.num_params));
    for (double& param : params) {
      param = readBinary<double>(file);
    }

    cameras[camera_id] =
        CameraRecord{camera_id, intrinsicsFromParams(model.name, width, height,
                                                     params)};
  }
  return cameras;
}

std::vector<ImageRecord> readImagesBinary(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP images binary file: " +
                             path.string());
  }

  const std::uint64_t num_images = readBinary<std::uint64_t>(file);
  std::vector<ImageRecord> images;
  images.reserve(static_cast<std::size_t>(num_images));
  for (std::uint64_t idx = 0U; idx < num_images; ++idx) {
    const std::uint32_t image_id = readBinary<std::uint32_t>(file);
    const double qw = readBinary<double>(file);
    const double qx = readBinary<double>(file);
    const double qy = readBinary<double>(file);
    const double qz = readBinary<double>(file);
    const double tx = readBinary<double>(file);
    const double ty = readBinary<double>(file);
    const double tz = readBinary<double>(file);
    const std::uint32_t camera_id = readBinary<std::uint32_t>(file);
    const std::string image_name = readNullTerminatedString(file);
    const std::uint64_t num_points2d = readBinary<std::uint64_t>(file);

    for (std::uint64_t point_idx = 0U; point_idx < num_points2d; ++point_idx) {
      readBinary<double>(file);
      readBinary<double>(file);
      readBinary<std::int64_t>(file);
    }

    images.push_back(ImageRecord{
        image_id, camera_id, image_name,
        worldFromCameraTransform(qw, qx, qy, qz, tx, ty, tz)});
  }
  return images;
}

std::unordered_map<std::string, fs::path> indexDepthFiles(
    const fs::path& depth_dir) {
  std::unordered_map<std::string, fs::path> files;
  if (!fs::exists(depth_dir)) {
    return files;
  }
  for (const fs::directory_entry& entry :
       fs::recursive_directory_iterator(depth_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    files.emplace(entry.path().filename().string(), entry.path());
  }
  return files;
}

std::vector<std::string> depthSuffixes(const bool prefer_geometric_depth) {
  if (prefer_geometric_depth) {
    return {".geometric.bin", ".photometric.bin"};
  }
  return {".photometric.bin", ".geometric.bin"};
}

bool isPngDepthPath(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](const unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return extension == ".png";
}

std::string findDepthPath(
    const fs::path& depth_dir, const std::string& image_name,
    const bool prefer_geometric_depth,
    const std::unordered_map<std::string, fs::path>& indexed_depth_files) {
  const std::vector<std::string> suffixes =
      depthSuffixes(prefer_geometric_depth);
  const std::string basename = fs::path(image_name).filename().string();

  for (const std::string& suffix : suffixes) {
    const fs::path direct_path = depth_dir / (image_name + suffix);
    if (fs::exists(direct_path)) {
      return direct_path.string();
    }
    const fs::path basename_path = depth_dir / (basename + suffix);
    if (fs::exists(basename_path)) {
      return basename_path.string();
    }
  }

  for (const std::string& suffix : suffixes) {
    const auto exact = indexed_depth_files.find(image_name + suffix);
    if (exact != indexed_depth_files.end()) {
      return exact->second.string();
    }
    const auto by_basename = indexed_depth_files.find(basename + suffix);
    if (by_basename != indexed_depth_files.end()) {
      return by_basename->second.string();
    }
  }

  // Rendered depth datasets commonly store a 16-bit PNG using the RGB image
  // stem (for example DSC01469.JPG -> DSC01469.png).
  const fs::path image_path(image_name);
  const fs::path png_relative_path =
      fs::path(image_path).replace_extension(".png");
  const fs::path png_basename =
      fs::path(basename).replace_extension(".png");
  for (const fs::path& png_name : {png_relative_path, png_basename}) {
    const fs::path direct_path = depth_dir / png_name;
    if (fs::exists(direct_path)) {
      return direct_path.string();
    }
  }

  const auto png_by_basename =
      indexed_depth_files.find(png_basename.filename().string());
  if (png_by_basename != indexed_depth_files.end()) {
    return png_by_basename->second.string();
  }

  return "";
}

DepthImage loadColmapBinaryDepthImage(const std::string& depth_path,
                                      const float depth_scale_m) {
  std::ifstream file(depth_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP depth map: " + depth_path);
  }

  std::array<std::string, 3> header_fields;
  int delimiter_count = 0;
  char character = '\0';
  while (file.get(character)) {
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

  const int width = std::stoi(header_fields[0]);
  const int height = std::stoi(header_fields[1]);
  const int channels = std::stoi(header_fields[2]);
  if (width <= 0 || height <= 0 || channels <= 0) {
    throw std::runtime_error("Invalid COLMAP depth map dimensions: " +
                             depth_path);
  }

  const std::size_t pixel_count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t value_count =
      pixel_count * static_cast<std::size_t>(channels);
  std::vector<float> raw_values(value_count);
  file.read(reinterpret_cast<char*>(raw_values.data()),
            static_cast<std::streamsize>(value_count * sizeof(float)));
  if (!file) {
    throw std::runtime_error("COLMAP depth map data is truncated: " +
                             depth_path);
  }

  DepthImage image;
  image.width = width;
  image.height = height;
  image.depths_m.resize(pixel_count);
  for (std::size_t idx = 0U; idx < pixel_count; ++idx) {
    float depth = raw_values[idx] * depth_scale_m;
    if (!std::isfinite(depth) || depth <= 0.0f) {
      depth = 0.0f;
    }
    image.depths_m[idx] = depth;
  }
  return image;
}

DepthImage loadPngDepthImage(const std::string& depth_path,
                             const float depth_scale_m) {
  if (stbi_is_16_bit(depth_path.c_str()) == 0) {
    throw std::runtime_error("PNG depth map must be 16-bit: " + depth_path);
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  using StbiImage = std::unique_ptr<stbi_us, decltype(&stbi_image_free)>;
  StbiImage raw_values(
      stbi_load_16(depth_path.c_str(), &width, &height, &channels, 1),
      &stbi_image_free);
  if (!raw_values) {
    const char* reason = stbi_failure_reason();
    throw std::runtime_error(
        "Failed to decode 16-bit PNG depth map: " + depth_path +
        (reason == nullptr ? std::string()
                           : " (" + std::string(reason) + ")"));
  }
  if (width <= 0 || height <= 0 || channels != 1) {
    throw std::runtime_error(
        "PNG depth map must contain one grayscale channel: " + depth_path);
  }

  DepthImage image;
  image.width = width;
  image.height = height;
  const std::size_t pixel_count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  image.depths_m.resize(pixel_count);
  for (std::size_t idx = 0U; idx < pixel_count; ++idx) {
    image.depths_m[idx] =
        static_cast<float>(raw_values.get()[idx]) * depth_scale_m;
  }
  return image;
}

void readColmapSparse(const fs::path& sparse_path,
                      std::unordered_map<std::uint32_t, CameraRecord>* cameras,
                      std::vector<ImageRecord>* images) {
  const fs::path cameras_text = sparse_path / "cameras.txt";
  const fs::path images_text = sparse_path / "images.txt";
  const fs::path cameras_bin = sparse_path / "cameras.bin";
  const fs::path images_bin = sparse_path / "images.bin";

  if (fs::exists(cameras_text) && fs::exists(images_text)) {
    *cameras = readCamerasText(cameras_text);
    *images = readImagesText(images_text);
    return;
  }

  if (fs::exists(cameras_bin) && fs::exists(images_bin)) {
    *cameras = readCamerasBinary(cameras_bin);
    *images = readImagesBinary(images_bin);
    return;
  }

  throw std::runtime_error(
      "COLMAP sparse path must contain cameras/images as .txt or .bin: " +
      sparse_path.string());
}

}  // namespace

Dataset DatasetReader::loadColmapDataset(
    const ColmapDatasetOptions& options) {
  const fs::path sparse_path(options.sparse_path);
  const fs::path depth_dir(options.depth_dir);
  if (!fs::exists(sparse_path)) {
    throw std::runtime_error("COLMAP sparse path does not exist: " +
                             sparse_path.string());
  }
  if (!fs::exists(depth_dir)) {
    throw std::runtime_error("Depth map directory does not exist: " +
                             depth_dir.string());
  }

  std::unordered_map<std::uint32_t, CameraRecord> cameras;
  std::vector<ImageRecord> images;
  readColmapSparse(sparse_path, &cameras, &images);
  std::sort(images.begin(), images.end(),
            [](const ImageRecord& lhs, const ImageRecord& rhs) {
              return lhs.image_id < rhs.image_id;
            });

  const std::unordered_map<std::string, fs::path> indexed_depth_files =
      indexDepthFiles(depth_dir);

  Dataset dataset;
  for (const ImageRecord& image : images) {
    const auto camera_it = cameras.find(image.camera_id);
    if (camera_it == cameras.end()) {
      throw std::runtime_error("COLMAP image references missing camera id: " +
                               std::to_string(image.camera_id));
    }

    const std::string depth_path =
        findDepthPath(depth_dir, image.image_name, options.prefer_geometric_depth,
                      indexed_depth_files);
    if (depth_path.empty()) {
      continue;
    }

    dataset.frames.push_back(FrameInfo{
        static_cast<std::size_t>(image.image_id),
        image.image_name,
        depth_path,
        camera_it->second.intrinsics,
        image.t_w_c});
  }

  if (dataset.frames.empty()) {
    throw std::runtime_error(
        "No COLMAP images could be matched to depth maps in: " +
        depth_dir.string());
  }

  const bool png_depths = isPngDepthPath(dataset.frames.front().depth_path);
  for (const FrameInfo& frame : dataset.frames) {
    if (isPngDepthPath(frame.depth_path) != png_depths) {
      throw std::runtime_error(
          "A dataset cannot mix 16-bit PNG and COLMAP binary depth maps.");
    }
  }
  // ScanNet++ rendered PNGs store uint16 millimeters; COLMAP binary depth maps
  // already store float32 meters.
  dataset.depth_scale_m = png_depths ? 0.001f : 1.0f;
  return dataset;
}

DepthImage DatasetReader::loadDepthImage(const std::string& depth_path,
                                         const float depth_scale_m) {
  if (isPngDepthPath(depth_path)) {
    return loadPngDepthImage(depth_path, depth_scale_m);
  }
  return loadColmapBinaryDepthImage(depth_path, depth_scale_m);
}

CameraIntrinsics scaledCameraForDepth(const CameraIntrinsics& camera,
                                      const DepthImage& depth_image) {
  CameraIntrinsics scaled_camera = camera;
  if (camera.width == depth_image.width && camera.height == depth_image.height) {
    return scaled_camera;
  }

  const float scale_x =
      static_cast<float>(depth_image.width) / static_cast<float>(camera.width);
  const float scale_y =
      static_cast<float>(depth_image.height) / static_cast<float>(camera.height);
  scaled_camera.fx *= scale_x;
  scaled_camera.fy *= scale_y;
  scaled_camera.cx *= scale_x;
  scaled_camera.cy *= scale_y;
  scaled_camera.width = depth_image.width;
  scaled_camera.height = depth_image.height;
  return scaled_camera;
}

}  // namespace nvblox_gpu_sdf
