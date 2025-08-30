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
#ifndef NVCLUSTERLOD_COMMON_H
#define NVCLUSTERLOD_COMMON_H

#define NVCLUSTERLOD_VERSION 3

#include <math.h>
#include <nvcluster/nvcluster.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(NVCLUSTERLOD_BUILDER_SHARED)
#if defined(_MSC_VER)
// msvc
#if defined(NVCLUSTERLOD_BUILDER_COMPILING)
#define NVCLUSTERLOD_API __declspec(dllexport)
#else
#define NVCLUSTERLOD_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
// gcc/clang
#define NVCLUSTERLOD_API __attribute__((visibility("default")))
#else
// Unsupported. If hit, use cmake GenerateExportHeader
#pragma warning Unsupported compiler
#define NVCLUSTERLOD_API
#endif
#else  // defined(NVCLUSTERLOD_BUILDER_SHARED)
// static lib, no export needed
#define NVCLUSTERLOD_API
#endif

#ifdef __cplusplus
#define NVCLUSTERLOD_DEFAULT(x) = x
#define NVCLUSTERLOD_STATIC_ASSERT(cond) static_assert(cond);
#else
#define NVCLUSTERLOD_DEFAULT(x)
#define NVCLUSTERLOD_STATIC_ASSERT(cond)
#endif

typedef struct nvclusterlod_Vec3u
{
  uint32_t x NVCLUSTERLOD_DEFAULT(0u);
  uint32_t y NVCLUSTERLOD_DEFAULT(0u);
  uint32_t z NVCLUSTERLOD_DEFAULT(0u);
} nvclusterlod_Vec3u;

#define nvclusterlod_defaultVec3u() {0u, 0u, 0u}

typedef struct nvclusterlod_Sphere
{
  nvcluster_Vec3f center NVCLUSTERLOD_DEFAULT(nvcluster_defaultVec3f());
  float radius           NVCLUSTERLOD_DEFAULT(0.0f);
} nvclusterlod_Sphere;

NVCLUSTERLOD_STATIC_ASSERT(sizeof(nvclusterlod_Sphere) == 16)

// Returns an approximate error to distance ratio (i.e. asin(angularError)) for
// the center of a perspective projection a target pixel error, field of view,
// and resolution. This utility can be used to pre-compute a target LOD
// threshold.
inline float nvclusterlodErrorOverDistance(float errorSizeInPixels, float fov, float resolution)
{
  return sinf(atanf(tanf(fov * 0.5f) * errorSizeInPixels / resolution));
}

// Inverse of nvclusterlodErrorOverDistance.
inline float nvclusterlodPixelError(float quadricErrorOverDistance, float fov, float resolution)
{
  return tanf(asinf(quadricErrorOverDistance)) * resolution / tanf(fov * 0.5f);
}

typedef struct nvclusterlod_ContextCreateInfo
{
  // Version expected. nvclusterlodCreateContext() returns
  // nvclusterlod_Result::NVCLUSTERLOD_ERROR_CONTEXT_VERSION_MISMATCH if another
  // is found at runtime.
  uint32_t version NVCLUSTERLOD_DEFAULT(NVCLUSTERLOD_VERSION);

  // Set to NVCLUSTER_TRUE or NVCLUSTER_FALSE to enable or disable internal
  // parallelisation using std execution policies at runtime
  nvcluster_Bool parallelize NVCLUSTER_DEFAULT(NVCLUSTER_TRUE);

  // Cluster builder context for the LOD builder to use
  nvcluster_Context clusterContext NVCLUSTERLOD_DEFAULT(nullptr);
} nvclusterlod_ContextCreateInfo;

struct nvclusterlod_Context_t;
typedef struct nvclusterlod_Context_t* nvclusterlod_Context;

typedef enum nvclusterlod_Result
{
  NVCLUSTERLOD_SUCCESS = 0,
  NVCLUSTERLOD_ERROR_EMPTY_CLUSTER_GENERATING_GROUPS,
  NVCLUSTERLOD_ERROR_CLUSTERING_TRIANGLES_FAILED,
  NVCLUSTERLOD_ERROR_CLUSTERING_CLUSTERS_FAILED,
  NVCLUSTERLOD_ERROR_CLUSTERING_NODES_FAILED,
  NVCLUSTERLOD_ERROR_NODES_OVERFLOW,
  NVCLUSTERLOD_ERROR_EMPTY_LOD_LEVELS,
  NVCLUSTERLOD_ERROR_LOD_LEVELS_OVERFLOW,
  NVCLUSTERLOD_ERROR_CLUSTER_COUNT_NOT_DECREASING,  // infinite loop detection in iterative decimation
  NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW,
  NVCLUSTERLOD_ERROR_OUTPUT_INCONSISTENT_COUNTS,  // internal consistency
  NVCLUSTERLOD_ERROR_EMPTY_ROOT_CLUSTER,
  NVCLUSTERLOD_ERROR_PRODUCED_NAN_BOUNDING_SPHERES,
  NVCLUSTERLOD_ERROR_GROUP_CLUSTER_COUNT_OVERFLOW,
  NVCLUSTERLOD_ERROR_NODE_CHILD_COUNT_OVERFLOW,
  NVCLUSTERLOD_ERROR_NULL_INPUT,
  NVCLUSTERLOD_ERROR_CONTEXT_VERSION_MISMATCH,
  NVCLUSTERLOD_ERROR_CLUSTER_ITEM_VERTEX_COUNT_NOT_THREE,
  NVCLUSTERLOD_ERROR_MAKE_BOUNDING_SPHERES_FROM_EMPTY_SET,
} nvclusterlod_Result;

NVCLUSTERLOD_API uint32_t            nvclusterlodVersion(void);
NVCLUSTERLOD_API nvclusterlod_Result nvclusterlodCreateContext(const nvclusterlod_ContextCreateInfo* createInfo,
                                                               nvclusterlod_Context*                 context);
NVCLUSTERLOD_API nvclusterlod_Result nvclusterlodDestroyContext(nvclusterlod_Context context);
NVCLUSTERLOD_API const char*         nvclusterlodResultString(nvclusterlod_Result result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NVCLUSTERLOD_COMMON_H
