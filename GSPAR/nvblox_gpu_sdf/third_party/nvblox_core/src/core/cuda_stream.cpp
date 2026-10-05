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

#include "nvblox/core/cuda_stream.h"

#include "glog/logging.h"

#include "nvblox/core/internal/error_check.h"

namespace nvblox {

void CudaStreamAsync::synchronize() const {
  checkCudaErrors(cudaStreamSynchronize(*stream_ptr_));
}

CudaStreamOwning::~CudaStreamOwning() {
  this->synchronize();
  checkCudaErrors(cudaStreamDestroy(stream_));
}

CudaStreamOwning::CudaStreamOwning(const unsigned int flags)
    : CudaStreamAsync(&stream_) {
  checkCudaErrors(cudaStreamCreateWithFlags(&stream_, flags));
}

void DefaultStream::synchronize() const {
  checkCudaErrors(cudaStreamSynchronize(default_stream_));
}

std::shared_ptr<CudaStream> CudaStream::createCudaStream(
    CudaStreamType stream_type) {
  switch (stream_type) {
    case CudaStreamType::kLegacyDefault:
      return std::make_shared<DefaultStream>(cudaStreamLegacy);
    case CudaStreamType::kBlocking:
      return std::make_shared<CudaStreamOwning>(cudaStreamDefault);
    case CudaStreamType::kNonBlocking:
      return std::make_shared<CudaStreamOwning>(cudaStreamNonBlocking);
    case CudaStreamType::kPerThreadDefault:
      return std::make_shared<DefaultStream>(cudaStreamPerThread);
    default:
      throw std::invalid_argument("received unspported CudaStreamType!");
  }
}

}  // namespace nvblox
