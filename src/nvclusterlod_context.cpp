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


#include "nvclusterlod_context.hpp"

uint32_t nvclusterlodVersion(void)
{
  return NVCLUSTERLOD_VERSION;
}

nvclusterlod_Result nvclusterlodCreateContext(const nvclusterlod_ContextCreateInfo* createInfo, nvclusterlod_Context* context)
{
  if(createInfo == nullptr || context == nullptr)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INVALID_ARGUMENT;
  }
  if(createInfo->version != NVCLUSTERLOD_VERSION)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CONTEXT_VERSION_MISMATCH;
  }

  *context = new nvclusterlod_Context_t{createInfo->clusterContext};
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

nvclusterlod_Result nvclusterlodDestroyContext(nvclusterlod_Context context)
{
  if(context == nullptr)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INVALID_ARGUMENT;
  }
  delete context;
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

const char* nvclusterlodResultString(nvclusterlod_Result result)
{
  // clang-format off
  switch(result)
  {
    case nvclusterlod_Result::NVCLUSTERLOD_SUCCESS: return "NVCLUSTERLOD_SUCCESS";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_EMPTY_CLUSTER_GENERATING_GROUPS: return "NVCLUSTERLOD_ERROR_EMPTY_CLUSTER_GENERATING_GROUPS";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTERING_FAILED: return "NVCLUSTERLOD_ERROR_CLUSTERING_FAILED";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_NODE_OVERFLOW: return "NVCLUSTERLOD_ERROR_NODE_OVERFLOW";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_LOD_OVERFLOW: return "NVCLUSTERLOD_ERROR_LOD_OVERFLOW";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_COUNT_NOT_DECREASING: return "NVCLUSTERLOD_ERROR_CLUSTER_COUNT_NOT_DECREASING";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_GENERATING_GROUPS: return "NVCLUSTERLOD_ERROR_INCONSISTENT_GENERATING_GROUPS";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_ADJACENCY_GENERATION_FAILED: return "NVCLUSTERLOD_ERROR_ADJACENCY_GENERATION_FAILED";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW: return "NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_GENERATING_GROUPS_MISMATCH: return "NVCLUSTERLOD_ERROR_CLUSTER_GENERATING_GROUPS_MISMATCH";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_EMPTY_ROOT_CLUSTER: return "NVCLUSTERLOD_ERROR_EMPTY_ROOT_CLUSTER";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_BOUNDING_SPHERES: return "NVCLUSTERLOD_ERROR_INCONSISTENT_BOUNDING_SPHERES";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED: return "NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_INVALID_ARGUMENT: return "NVCLUSTERLOD_ERROR_INVALID_ARGUMENT";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_UNSPECIFIED: return "NVCLUSTERLOD_ERROR_UNSPECIFIED";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_CONTEXT_VERSION_MISMATCH: return "NVCLUSTERLOD_ERROR_CONTEXT_VERSION_MISMATCH";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_ITEM_VERTEX_COUNT_NOT_THREE: return "NVCLUSTERLOD_ERROR_CLUSTER_ITEM_VERTEX_COUNT_NOT_THREE";
    case nvclusterlod_Result::NVCLUSTERLOD_ERROR_MAKE_BOUNDING_SPHERES_FROM_EMPTY_SET: return "NVCLUSTERLOD_ERROR_MAKE_BOUNDING_SPHERES_FROM_EMPTY_SET";
    default: return "<Invalid nvclusterlod_Result>";
  }
  // clang-format on
}
