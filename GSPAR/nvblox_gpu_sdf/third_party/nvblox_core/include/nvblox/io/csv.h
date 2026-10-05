/*
Copyright 2022 NVIDIA CORPORATION

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

#include <string>

#include <Eigen/Eigen>

namespace nvblox {
namespace io {

/// Returns a path: /{output_dir}/{prefix}{idx}.{ext}
/// @param dir The output directory.
/// @param prefix The part of the filename before the index.
/// @param idx The index.
/// @param num_digits The number of digits in the idx string, preceeded by 0s.
/// @param ext The file extension.
/// @return The file path.
inline std::string getIndexedPath(const std::string& dir,
                                  const std::string& prefix, const int idx,
                                  const int num_digits = 4,
                                  const std::string& ext = ".txt");

template <typename Derived>
void writeToCsv(const std::string& filepath,
                const Eigen::DenseBase<Derived>& eig);

}  // namespace io
}  // namespace nvblox

#include "nvblox/io/internal/impl/csv_impl.h"
