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

#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/params.h"

namespace nvblox {

/// Takes a list of bounding boxes and produces a mask image.
/// We filter the bounding boxes around a histogram-computed modal depth. Only
/// pixels inside a bounding box with a depth within "mode_proximity_threshold"
/// are included in the output mask image.
/// @param detections A list of bounding box detections.
/// @param depth_image A depth image used to filter the bounding boxes.
/// @param mode_proximity_threshold The distance passing pixels must be from the
/// modal depth to be included in the mask.
/// @param mask The output mask image.
/// @param cuda_stream A stream on which to process the work.
void maskFromDetections(const std::vector<ImageBoundingBox>& detections,
                        const DepthImage& depth_image,
                        const float mode_proximity_threshold, MonoImage* mask,
                        const CudaStream& cuda_stream);

/// Clips a bounding box to be within the size of the image.
/// @param detection The bounding box.
/// @param rows Number of rows in the image.
/// @param cols Number of cols in the image.
/// @return The clipped bounding box.
ImageBoundingBox clipToImageBounds(const ImageBoundingBox& detection,
                                   const int rows, const int cols);

}  // namespace nvblox
