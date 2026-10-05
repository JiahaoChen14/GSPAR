/*
Copyright 2024i NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
kWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
Skee the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "nvblox/serialization/layer_serializer_gpu.h"
#include "nvblox/serialization/mesh_serializer_gpu.h"

namespace nvblox {
// The traits defined here are used to deduce the correct SerializerType and
// SerializedLayerType given for a given LayerType. This allow us to template
// LayerStreamer on LayerType only. End effect is less code bloat and
// easier integration into compound structures such as the LayerCake.

// Type traits for determining the correct serializer given a layer.
template <typename LayerType>
struct SerializerTypeTrait {
  struct NoSerializerAvailable {
    struct NoSerializedLayerAvailable {};
    using SerializedLayerType = NoSerializedLayerAvailable;
  };
  using type = NoSerializerAvailable;
};

template <>
struct SerializerTypeTrait<MeshLayer> {
  using type = MeshSerializerGpu;
};

template <>
struct SerializerTypeTrait<TsdfLayer> {
  using type = TsdfLayerSerializerGpu;
};

template <>
struct SerializerTypeTrait<EsdfLayer> {
  using type = EsdfLayerSerializerGpu;
};

template <>
struct SerializerTypeTrait<OccupancyLayer> {
  using type = OccupancyLayerSerializerGpu;
};

template <>
struct SerializerTypeTrait<FreespaceLayer> {
  using type = FreespaceLayerSerializerGpu;
};

template <>
struct SerializerTypeTrait<ColorLayer> {
  using type = ColorLayerSerializerGpu;
};

struct DummyLayer;
template <>
struct SerializerTypeTrait<DummyLayer> {
  using type = FreespaceLayerSerializerGpu;
};

template <typename LayerType>
using SerializerType = typename SerializerTypeTrait<LayerType>::type;

// Type traits for determining the correct serialized output type given a
// layer.
template <typename LayerType>
struct SerializedLayerTypeTrait {
  using type = typename SerializerType<LayerType>::SerializedLayerType;
};

template <typename LayerType>
using SerializedLayerType = typename SerializedLayerTypeTrait<LayerType>::type;
}  // namespace nvblox
