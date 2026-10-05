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

#include <bitset>
#include <memory>

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/npp_image_operations.h"
#include "nvblox/utils/logging.h"

namespace nvblox {
namespace image {

/// Remove small connected components from mask image
class MaskPreprocessor {
 public:
  MaskPreprocessor(std::shared_ptr<CudaStream> cuda_stream);

  /// Remove small connected components from mask image
  ///
  /// @attention Resulting mask will be non-zero for active pixels, but not
  ///            necessarily 255.
  ///
  /// @param mask            Target image. Any non-zero pixels are masked
  /// @param size_threshold  Keep only components with num pixels more than this
  /// @param mask_out        Output image
  void removeSmallConnectedComponents(const MonoImage& mask,
                                      const int size_threshold,
                                      MonoImage* mask_out);

 private:
  // To decrease runtime, the incoming mask image will be downsampled with this
  // factor before processing.
  static constexpr int kDownScaleFactor = 2;

  // buffers for temporary storage
  MonoImage mask_downscaled_{MemoryType::kDevice};
  MonoImage mask_thresholded_host_{MemoryType::kHost};
  MonoImage mask_upscaled_{MemoryType::kDevice};

  // Cuda stream and context state
  std::shared_ptr<CudaStream> cuda_stream_;
  NppStreamContext npp_stream_context_;
};

}  // namespace image
};  // namespace nvblox
