/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <nvcluster/nvcluster.h>

#define NVCLUSTERLOD_VERSION 1

namespace nvclusterlod {

struct Sphere
{
  float x      = 0.f;
  float y      = 0.f;
  float z      = 0.f;
  float radius = 0.f;
};

inline float pixelErrorToQuadricErrorOverDistance(float errorSizeInPixels, float fov, float resolution)
{
  return sinf(atanf(tanf(fov * 0.5f) * errorSizeInPixels / resolution));
}

inline float quadricErrorOverDistanceToPixelError(float quadricErrorOverDistance, float fov, float resolution)
{
  return tanf(asinf(quadricErrorOverDistance)) * resolution / tanf(fov * 0.5f);
}

struct ContextCreateInfo
{
  uint32_t           version        = NVCLUSTERLOD_VERSION;
  nvcluster::Context clusterContext = nullptr;
};

struct Context_t;
typedef Context_t* Context;

enum Result
{
  SUCCESS = 0,
  ERROR_EMPTY_CLUSTER_GENERATING_GROUPS,
  ERROR_CLUSTERING_FAILED,
  ERROR_NODE_OVERFLOW,
  ERROR_LOD_OVERFLOW,
  ERROR_CLUSTER_COUNT_NOT_DECREASING,
  ERROR_INCONSISTENT_GENERATING_GROUPS,
  ERROR_ADJACENCY_GENERATION_FAILED,
  ERROR_OUTPUT_MESH_OVERFLOW,
  ERROR_CLUSTER_GENERATING_GROUPS_MISMATCH,
  ERROR_EMPTY_ROOT_CLUSTER,
  ERROR_INCONSISTENT_BOUNDING_SPHERES,
  ERROR_HIERARCHY_GENERATION_FAILED,
  ERROR_INVALID_ARGUMENT,
  ERROR_UNSPECIFIED,
};

}  // namespace nvclusterlod


extern "C" {


nvclusterlod::Result nvclusterlodCreateContext(const nvclusterlod::ContextCreateInfo* createInfo, nvclusterlod::Context* context);
nvclusterlod::Result nvclusterlodDestroyContext(nvclusterlod::Context context);
}
