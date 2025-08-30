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

#include <cassert>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvclusterlod {

// Shortcut and storage for LOD output
struct LodMesh
{
  template <class Counts>
  void resize(const Counts& counts)
  {
    triangleVertices.resize(counts.triangleCount);
    clusterTriangleRanges.resize(counts.clusterCount);
    clusterGeneratingGroups.resize(counts.clusterCount);
    clusterBoundingSpheres.resize(counts.clusterCount);
    groupQuadricErrors.resize(counts.groupCount);
    groupClusterRanges.resize(counts.groupCount);
    lodLevelGroupRanges.resize(counts.lodLevelCount);
  }

  void shrink_to_fit()
  {
    // If keeping the object around, reallocating the conservatively sized
    // output memory is worthwhile.
    triangleVertices.shrink_to_fit();
    clusterTriangleRanges.shrink_to_fit();
    clusterGeneratingGroups.shrink_to_fit();
    clusterBoundingSpheres.shrink_to_fit();
    groupQuadricErrors.shrink_to_fit();
    groupClusterRanges.shrink_to_fit();
    lodLevelGroupRanges.shrink_to_fit();
  }

  std::vector<nvclusterlod_Vec3u>  triangleVertices;
  std::vector<nvcluster_Range>     clusterTriangleRanges;
  std::vector<uint32_t>            clusterGeneratingGroups;
  std::vector<nvclusterlod_Sphere> clusterBoundingSpheres;
  std::vector<float>               groupQuadricErrors;
  std::vector<nvcluster_Range>     groupClusterRanges;
  std::vector<nvcluster_Range>     lodLevelGroupRanges;
};

// LodMesh delayed init constructor
inline nvclusterlod_Result generateLodMesh(nvclusterlod_Context context, const nvclusterlod_MeshInput& input, LodMesh& lodMesh)
{
  // Get conservative output sizes
  nvclusterlod_MeshCounts counts;
  if(nvclusterlod_Result result = nvclusterlodGetMeshRequirements(context, &input, &counts); result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }

  // Allocate storage
  lodMesh.resize(counts);

  // Make LODs
  nvclusterlod_MeshOutput lodOutput{};
  lodOutput.clusterTriangleRanges   = lodMesh.clusterTriangleRanges.data();
  lodOutput.triangleVertices        = lodMesh.triangleVertices.data();
  lodOutput.clusterGeneratingGroups = lodMesh.clusterGeneratingGroups.data();
  lodOutput.clusterBoundingSpheres  = lodMesh.clusterBoundingSpheres.data();
  lodOutput.groupQuadricErrors      = lodMesh.groupQuadricErrors.data();
  lodOutput.groupClusterRanges      = lodMesh.groupClusterRanges.data();
  lodOutput.lodLevelGroupRanges     = lodMesh.lodLevelGroupRanges.data();
  lodOutput.clusterCount            = uint32_t(lodMesh.clusterTriangleRanges.size());
  lodOutput.groupCount              = uint32_t(lodMesh.groupQuadricErrors.size());
  lodOutput.lodLevelCount           = uint32_t(lodMesh.lodLevelGroupRanges.size());
  lodOutput.triangleCount           = uint32_t(lodMesh.triangleVertices.size());

  if(nvclusterlod_Result result = nvclusterlodBuildMesh(context, &input, &lodOutput); result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }

  // Truncate the output to the size written
  lodMesh.resize(lodOutput);
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

struct LocalizedLodMesh
{
  LodMesh                       lodMesh;  // contains cluster-local triangle indices
  std::vector<nvcluster_Range>  clusterVertexRanges;
  std::vector<uint32_t>         vertexGlobalIndices;

  // Per-cluster maximums
  uint32_t maxClusterTriangles = 0;
  uint32_t maxClusterVertices  = 0;
};

// Computes unique triangle vertices per cluster, returning vertex ranges and
// their indices into the original global vertices. Localized triangle vertices
// are written to clusterTriangleVerticesLocal, allowing in-place conversion.
// TODO: parallelize?
// TODO: LocalizedClusterVertices::vertexGlobalIndices indirection is transient
//       and might better be handled as a callback
inline nvclusterlod_Result generateLocalizedLodMesh(LodMesh&& input, LocalizedLodMesh& localizedMesh)
{
  if(&localizedMesh.lodMesh != &input)
  {
    localizedMesh.lodMesh = std::move(input);
  }

  for(size_t clusterTriangleRangeIndex = 0;
      clusterTriangleRangeIndex < localizedMesh.lodMesh.clusterTriangleRanges.size(); clusterTriangleRangeIndex++)
  {
    const nvcluster_Range& clusterTriangleRange = localizedMesh.lodMesh.clusterTriangleRanges[clusterTriangleRangeIndex];
    std::span<const nvclusterlod_Vec3u> globalTriangles(
        localizedMesh.lodMesh.triangleVertices.data() + clusterTriangleRange.offset, clusterTriangleRange.count);
    std::span<nvclusterlod_Vec3u> localTriangles(localizedMesh.lodMesh.triangleVertices.data() + clusterTriangleRange.offset,
                                                 clusterTriangleRange.count);

    uint32_t currentLocalTriangleIndex = 0;

    nvcluster_Range vertexRange{.offset = uint32_t(localizedMesh.vertexGlobalIndices.size()), .count = 0};

    {
      std::unordered_map<uint32_t, uint32_t> vertexCache;
      for(size_t globalTriangleIndex = 0; globalTriangleIndex < globalTriangles.size(); globalTriangleIndex++)
      {
        const auto& inputTriangle  = reinterpret_cast<const uint32_t(&)[3]>(globalTriangles[globalTriangleIndex]);
        auto&       outputTriangle = reinterpret_cast<uint32_t(&)[3]>(localTriangles[currentLocalTriangleIndex]);
        currentLocalTriangleIndex++;
        for(int j = 0; j < 3; ++j)
        {
          auto [vertIndexIt, isNew] = vertexCache.try_emplace(inputTriangle[j], uint32_t(vertexCache.size()));

          if(isNew)
          {
            localizedMesh.vertexGlobalIndices.push_back(inputTriangle[j]);
          }
          outputTriangle[j] = vertIndexIt->second;
        }
      }
      vertexRange.count = uint32_t(vertexCache.size());
    }
    localizedMesh.clusterVertexRanges.push_back(vertexRange);
    localizedMesh.maxClusterTriangles = std::max(localizedMesh.maxClusterTriangles, clusterTriangleRange.count);
    localizedMesh.maxClusterVertices  = std::max(localizedMesh.maxClusterVertices, vertexRange.count);
  }
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

inline nvclusterlod_Result generateLocalizedLodMesh(nvclusterlod_Context context, const nvclusterlod_MeshInput& input, LocalizedLodMesh& localizedMesh)
{
  LodMesh lodMesh;
  if(nvclusterlod_Result result = generateLodMesh(context, input, lodMesh); result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }
  return generateLocalizedLodMesh(std::move(lodMesh), localizedMesh);
}

// Utility call to build lists of generating groups (that contributed
// decimated clusters) for each group. This collapses duplicate values in
// clusterGeneratingGroups for each groupClusterRanges.
struct GroupGeneratingGroups
{
  std::vector<nvcluster_Range>  ranges;  // ranges of groups
  std::vector<uint32_t>         groups;  // indices of generating groups

  // Accessors to view this struct as an array of arrays. This avoids having the
  // many heap allocations that a std::vector of vectors has.
  std::span<const uint32_t> operator[](size_t i) const
  {
    return std::span(groups.data() + ranges[i].offset, ranges[i].count);
  }
  size_t size() const { return ranges.size(); }
};

inline nvclusterlod_Result generateGroupGeneratingGroups(std::span<const nvcluster_Range>     groupClusterRanges,
                                                         std::span<const uint32_t>            clusterGeneratingGroups,
                                                         nvclusterlod::GroupGeneratingGroups& groupGeneratingGroups)
{
  groupGeneratingGroups.ranges.reserve(groupClusterRanges.size());

  // Iterate over all groups, find the unique set of generating groups from
  // their clusters and append them linearly
  for(size_t groupIndex = 0; groupIndex < groupClusterRanges.size(); groupIndex++)
  {
    const nvcluster_Range& clusterRange = groupClusterRanges[groupIndex];
    if(clusterRange.count == 0)
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_EMPTY_CLUSTER_GENERATING_GROUPS;
    }

    std::span<const uint32_t> generatingGroups =
        std::span(clusterGeneratingGroups).subspan(clusterRange.offset, clusterRange.count);

    if(generatingGroups[0] == NVCLUSTERLOD_ORIGINAL_MESH_GROUP)
    {
      groupGeneratingGroups.ranges.push_back({uint32_t(groupGeneratingGroups.groups.size()), 0});  // LOD0 groups have no generating group
    }
    else
    {
      std::unordered_set uniqueGeneratingGroups(generatingGroups.begin(), generatingGroups.end());
      nvcluster_Range newGroupRange = {uint32_t(groupGeneratingGroups.groups.size()), uint32_t(uniqueGeneratingGroups.size())};
      groupGeneratingGroups.ranges.push_back(newGroupRange);
      groupGeneratingGroups.groups.insert(groupGeneratingGroups.groups.end(), uniqueGeneratingGroups.begin(),
                                          uniqueGeneratingGroups.end());
    }
  }
  assert(groupGeneratingGroups.ranges[0].offset == 0);
  assert(groupGeneratingGroups.ranges.size() == groupClusterRanges.size());
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}
}  // namespace nvclusterlod
