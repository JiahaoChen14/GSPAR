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

#include <cuda_runtime.h>
#include <npp.h>

namespace nvblox {

/// Allow for NVBLOX_CHECK(expression) in both host and device code.
/// Halts execution if expression evaluates to false. Unlike assert, these
/// checks will remain active for RELEASE builds and should therefore be used
/// sparingly to avoid overhead.
#ifdef __CUDA_ARCH__
#define NVBLOX_CHECK(expression, message_string)                 \
  {                                                              \
    if (!(expression)) {                                         \
      printf(                                                    \
          "%s:%i (function: %s)  ERROR: CUDA check failed with " \
          "message\n%s\n\n",                                     \
          __FILE__, __LINE__, __FUNCTION__, message_string);     \
      __trap();                                                  \
    }                                                            \
  }
#else
#define NVBLOX_CHECK(expression, message_string) \
  CHECK(expression) << message_string;
#endif

// Debug-only macros that will be disabled for release builds.
#ifdef NDEBUG
#define NVBLOX_DCHECK(expression, message_string)
#else
#define NVBLOX_DCHECK(expression, message_string) \
  NVBLOX_CHECK(expression, message_string)
#endif

/// Aborts execution with a message
#define NVBLOX_ABORT(message_string) NVBLOX_CHECK(false, message_string);

void check_cuda_error_value(cudaError_t result, char const* const func,
                            const char* const file, int const line);

#define checkCudaErrors(val) \
  nvblox::check_cuda_error_value((val), #val, __FILE__, __LINE__)

void check_npp_error_value(NppStatus status, char const* const func,
                           const char* const file, int const line);

#define checkNppErrors(val) \
  nvblox::check_npp_error_value((val), #val, __FILE__, __LINE__)

}  // namespace nvblox
