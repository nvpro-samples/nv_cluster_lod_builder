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

#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nvclusterlod/nvclusterlod_mesh.h>


namespace nvclusterlod {


// Shortcut and storage for LOD output
struct LodMesh
{
  void resize(const MeshRequirements& sizes)
  {
    triangleVertices.resize(sizes.maxTriangleCount * 3);
    clusterTriangleRanges.resize(sizes.maxClusterCount);
    clusterGeneratingGroups.resize(sizes.maxClusterCount);
    clusterBoundingSpheres.resize(sizes.maxClusterCount);
    groupQuadricErrors.resize(sizes.maxGroupCount);
    groupClusterRanges.resize(sizes.maxGroupCount);
    lodLevelGroupRanges.resize(sizes.maxLodLevelCount);
  }

  std::vector<uint32_t>         triangleVertices;
  std::vector<nvcluster::Range> clusterTriangleRanges;
  std::vector<uint32_t>         clusterGeneratingGroups;
  std::vector<Sphere>           clusterBoundingSpheres;
  std::vector<float>            groupQuadricErrors;
  std::vector<nvcluster::Range> groupClusterRanges;
  std::vector<nvcluster::Range> lodLevelGroupRanges;
};


inline nvclusterlod::Result generateLodMesh(nvclusterlod::Context context, const MeshInput& input, LodMesh& lodMesh)
{
  MeshGetRequirementsInfo reqInfo;
  reqInfo.input = &input;

  // Get conservative output sizes
  MeshRequirements sizes = nvclusterlodMeshGetRequirements(context, &reqInfo);

  // Allocate storage
  lodMesh.resize(sizes);

  // Make LODs
  nvclusterlod::MeshOutput lodOutput{};
  lodOutput.clusterTriangleRanges   = lodMesh.clusterTriangleRanges.data();
  lodOutput.clusterTriangles        = reinterpret_cast<uint32_t*>(lodMesh.triangleVertices.data());
  lodOutput.clusterGeneratingGroups = lodMesh.clusterGeneratingGroups.data();
  lodOutput.clusterBoundingSpheres  = lodMesh.clusterBoundingSpheres.data();
  lodOutput.groupQuadricErrors      = lodMesh.groupQuadricErrors.data();
  lodOutput.groupClusterRanges      = lodMesh.groupClusterRanges.data();
  lodOutput.lodLevelGroupRanges     = lodMesh.lodLevelGroupRanges.data();
  lodOutput.clusterCount            = uint32_t(lodMesh.clusterTriangleRanges.size());
  lodOutput.groupCount              = uint32_t(lodMesh.groupQuadricErrors.size());
  lodOutput.lodLevelCount           = uint32_t(lodMesh.lodLevelGroupRanges.size());
  lodOutput.triangleCount           = uint32_t(lodMesh.triangleVertices.size());

  MeshCreateInfo createInfo;
  createInfo.input = &input;

  nvclusterlod::Result result = nvclusterlodMeshCreate(context, &createInfo, &lodOutput);

  if(result != nvclusterlod::Result::SUCCESS)
  {
    lodOutput = {};
    return result;
  }

  sizes.maxClusterCount  = lodOutput.clusterCount;
  sizes.maxGroupCount    = lodOutput.groupCount;
  sizes.maxLodLevelCount = lodOutput.lodLevelCount;
  sizes.maxTriangleCount = lodOutput.triangleCount;

  // Truncate to output size written
  lodMesh.resize(sizes);
  return nvclusterlod::Result::SUCCESS;
}


struct LocalizedLodMesh
{
  LodMesh                       lodMesh;  // contains cluster-local triangle indices
  std::vector<nvcluster::Range> clusterVertexRanges;
  std::vector<uint32_t>         vertexGlobalIndices;
};

// Computes unique triangle vertices per cluster, returning vertex ranges and
// their indices into the original global vertices. Localized triangle vertices
// are written to clusterTriangleVerticesLocal, allowing in-place conversion.
// TODO: parallelize?
// TODO: LocalizedClusterVertices::vertexGlobalIndices indirection is transient
//       and might better be handled as a callback
inline nvclusterlod::Result generateLocalizedLodMesh(nvclusterlod::Context context, const MeshInput& input, LocalizedLodMesh& localizedMesh)

{
  nvclusterlod::Result result = generateLodMesh(context, input, localizedMesh.lodMesh);
  if(result != nvclusterlod::Result::SUCCESS)
  {
    return result;
  }

  for(size_t clusterTriangleRangeIndex = 0;
      clusterTriangleRangeIndex < localizedMesh.lodMesh.clusterTriangleRanges.size(); clusterTriangleRangeIndex++)
  {

    const nvcluster::Range& clusterTriangleRange = localizedMesh.lodMesh.clusterTriangleRanges[clusterTriangleRangeIndex];

    // FIXME: the indices are changed from global to cluster-local in-place, this needs to be made clear in the doc
    std::span<const uint32_t> globalTriangles(localizedMesh.lodMesh.triangleVertices.data() + 3 * clusterTriangleRange.offset,
                                              clusterTriangleRange.count * 3);
    std::span<uint32_t> localTriangles(localizedMesh.lodMesh.triangleVertices.data() + 3 * clusterTriangleRange.offset,
                                       clusterTriangleRange.count * 3);

    uint32_t currentLocalTriangleIndex = 0;

    nvcluster::Range vertexRange{.offset = uint32_t(localizedMesh.vertexGlobalIndices.size()), .count = 0};

    {
      std::unordered_map<uint32_t, uint32_t> vertexCache;
      for(size_t globalTriangleIndex = 0; globalTriangleIndex < globalTriangles.size() / 3; globalTriangleIndex++)
      {
        const uint32_t* inputTriangle  = &globalTriangles[3 * globalTriangleIndex];
        uint32_t*       outputTriangle = &localTriangles[3 * currentLocalTriangleIndex];
        currentLocalTriangleIndex++;
        for(int j = 0; j < 3; ++j)
        {
          auto [vertIndexIt, isNew] = vertexCache.try_emplace(inputTriangle[j], uint32_t(vertexCache.size()));

          if(isNew)
          {
            localizedMesh.vertexGlobalIndices.push_back(inputTriangle[j]);
          }
          outputTriangle[j] = vertIndexIt->second;
          if(outputTriangle[j] >= 256)
          {
            return nvclusterlod::Result::ERROR_OUTPUT_MESH_OVERFLOW;
          }
        }
      }
      vertexRange.count = uint32_t(vertexCache.size());
    }
    localizedMesh.clusterVertexRanges.push_back(vertexRange);
  }
  return nvclusterlod::Result::SUCCESS;
}


// Utility call to build lists of generating groups (that contributed
// decimated clusters) for each group. This collapses duplicate values in
// clusterGeneratingGroups for each groupClusterRanges.
struct GroupGeneratingGroups
{
  std::vector<nvcluster::Range> ranges;  // ranges of groups
  std::vector<uint32_t>         groups;  // indices of generating groups

  // Accessors to view this struct as an array of arrays. This avoids having the
  // many heap allocations that a std::vector of vectors has.
  std::span<const uint32_t> operator[](size_t i) const
  {
    return std::span(groups.data() + ranges[i].offset, ranges[i].count);
  }
  size_t size() const { return ranges.size(); }
};


inline nvclusterlod::Result generateGroupGeneratingGroups(std::span<const nvcluster::Range>    groupClusterRanges,
                                                          std::span<const uint32_t>            clusterGeneratingGroups,
                                                          nvclusterlod::GroupGeneratingGroups& groupGeneratingGroups)
{
  groupGeneratingGroups.ranges.reserve(groupClusterRanges.size());

  // iterate over all groups and find unique set of generating groups
  // from its clusters.
  //
  // append them linearly

  uint32_t offset = 0;

  for(size_t groupIndex = 0; groupIndex < groupClusterRanges.size(); groupIndex++)
  {
    const nvcluster::Range& clusterRange = groupClusterRanges[groupIndex];
    if(clusterRange.count == 0)
    {
      return nvclusterlod::Result::ERROR_EMPTY_CLUSTER_GENERATING_GROUPS;
    }

    std::span<const uint32_t> generatingGroups(clusterGeneratingGroups.data() + clusterRange.offset, clusterRange.count);

    if(generatingGroups[0] == ORIGINAL_MESH_GROUP)
    {
      groupGeneratingGroups.ranges.push_back({offset, 0});  // LOD0 groups have no generating group
    }
    else
    {
      std::unordered_set uniqueGeneratingGroups(generatingGroups.begin(), generatingGroups.end());
      nvcluster::Range   newGroupRange = {offset, uint32_t(uniqueGeneratingGroups.size())};
      groupGeneratingGroups.ranges.push_back(newGroupRange);
      groupGeneratingGroups.groups.insert(groupGeneratingGroups.groups.end(), uniqueGeneratingGroups.begin(),
                                          uniqueGeneratingGroups.end());

      offset += newGroupRange.count;
    }
  }
  return nvclusterlod::Result::SUCCESS;
}
}  // namespace nvclusterlod
