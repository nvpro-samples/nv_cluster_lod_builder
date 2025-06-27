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

#include <vector>

#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>

namespace nvclusterlod {

// Shortcut and storage for hierarchy output
struct LodHierarchy
{
  std::vector<nvclusterlod_HierarchyNode> nodes;
  std::vector<nvclusterlod_Sphere>        groupCumulativeBoundingSpheres;
  std::vector<float>  groupCumulativeQuadricError;
};

inline nvclusterlod_Result generateLodHierarchy(nvclusterlod_Context context, const nvclusterlod_HierarchyInput& input, LodHierarchy& hierarchy)
{
  // Get conservative output sizes
  nvclusterlod_HierarchyCounts sizes;
  if(nvclusterlod_Result result = nvclusterlodGetHierarchyRequirements(context, &input, &sizes);
     result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }

  // Allocate storage
  hierarchy.nodes.resize(sizes.nodeCount);
  hierarchy.groupCumulativeBoundingSpheres.resize(input.groupCount);
  hierarchy.groupCumulativeQuadricError.resize(input.groupCount);

  // Pack output pointers
  nvclusterlod_HierarchyOutput output;
  output.groupCumulativeBoundingSpheres = hierarchy.groupCumulativeBoundingSpheres.data();
  output.groupCumulativeQuadricError    = hierarchy.groupCumulativeQuadricError.data();
  output.nodeCount                      = sizes.nodeCount;
  output.nodes                          = hierarchy.nodes.data();

  // Make LODs
  if(nvclusterlod_Result result = nvclusterlodBuildHierarchy(context, &input, &output); result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    hierarchy = {};
    return result;
  }
  // Truncate to output size written
  hierarchy.nodes.resize(output.nodeCount);
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

inline nvclusterlod_HierarchyInput makeHierarchyInput(const nvclusterlod_MeshOutput& meshOutput)
{
  return {
      .clusterGeneratingGroups = meshOutput.clusterGeneratingGroups,
      .clusterBoundingSpheres  = meshOutput.clusterBoundingSpheres,
      .groupQuadricErrors      = meshOutput.groupQuadricErrors,
      .groupClusterRanges      = meshOutput.groupClusterRanges,
      .lodLevelGroupRanges     = meshOutput.lodLevelGroupRanges,
      .clusterCount            = meshOutput.clusterCount,
      .groupCount              = meshOutput.groupCount,
      .lodLevelCount           = meshOutput.lodLevelCount,
  };
}

inline nvclusterlod_Result generateLodHierarchy(nvclusterlod_Context context, const nvclusterlod_MeshOutput& meshOutput, LodHierarchy& hierarchy)
{
  return generateLodHierarchy(context, makeHierarchyInput(meshOutput), hierarchy);
}

}  // namespace nvclusterlod
