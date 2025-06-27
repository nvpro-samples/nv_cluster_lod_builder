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

// Workaround for libc++ std::execution
#include "../nv_cluster_builder/src/parallel_execution_libcxx.hpp"

#include <execution>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod_context.hpp>
#include <nvclusterlod_parallel.hpp>
#include <nvclusterlod_util.hpp>
#include <span>
#include <vector>

// Create a 32-bit mask with the lowest bitCount bits set to 1.
// bitCount must be less than 32.
#define U32_MASK(bitCount) ((1u << (bitCount)) - 1u)

// From the set of input nodes, cluster them according to their spatial location so each cluster contains at most maxClusterItems
static nvcluster_Result clusterNodesSpatially(nvcluster_Context                           context,
                                              std::span<const nvclusterlod_HierarchyNode> nodes,
                                              uint32_t                                    maxClusterItems,
                                              nvcluster::ClusterStorage&                  clusters)
{
  // For each node, compute its axis-aligned bounding box and centroid location
  std::vector<nvcluster_AABB>  triangleClusterAabbs(nodes.size());
  std::vector<float>           triangleClusterCentroids(nodes.size() * 3);

  NVLOD_PARALLEL_FOR_BEGIN(nodeIndex, nodes.size(), 2048)
  {
    const nvclusterlod_HierarchyNode& node   = nodes[nodeIndex];
    auto                              center = fromAPI(node.boundingSphere.center);

    nvcluster_AABB& aabb = triangleClusterAabbs[nodeIndex];
    for(uint32_t i = 0; i < 3; i++)
    {
      aabb.bboxMin[i] = center[i] - node.boundingSphere.radius;
      aabb.bboxMax[i] = center[i] + node.boundingSphere.radius;
    }

    for(uint32_t i = 0; i < 3; i++)
    {
      triangleClusterCentroids[3 * nodeIndex + i] = (aabb.bboxMin[i] + aabb.bboxMax[i]) * 0.5f;
    }
  }
  NVLOD_PARALLEL_FOR_END;

  // Call the clusterizer to group the nodes
  nvcluster_Input clusterBounds{
      .itemBoundingBoxes = triangleClusterAabbs.data(),
      .itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(triangleClusterCentroids.data()),
      .itemCount         = uint32_t(triangleClusterAabbs.size()),
  };

  nvcluster_Config config = {
      .minClusterSize = maxClusterItems,
      .maxClusterSize = maxClusterItems,
  };

  nvcluster_Result result = nvcluster::generateClusters(context, config, clusterBounds, clusters);
  return result;
}

//union SphereDist
//{
//  struct
//  {
//    float dist;
//    uint32_t index;
//  };
//  std::atomic_uint64_t u64;
//};

// Find the sphere within spheres that lies farthest from the start sphere, accounting for the radii of the spheres
inline nvclusterlod_Sphere farthestSphere(std::span<const nvclusterlod_Sphere> spheres, const nvclusterlod_Sphere& start)
{
  nvclusterlod_Sphere  result    = start;
  float                maxLength = 0.0f;

  // FIXME: todo for parallelism
  //SphereDist sd{.u64 = 0ull};


  // FIXME: parallelize?
  //for(const nvlod::Sphere& candidate : spheres)
  //NVLOD_PARALLEL_FOR_BEGIN(sphereIndex, spheres.size())
  for(size_t sphereIndex = 0; sphereIndex < spheres.size(); sphereIndex++)
  {
    const nvclusterlod_Sphere& candidate               = spheres[sphereIndex];
    auto                       centerToCandidateVector = fromAPI(candidate.center) - fromAPI(start.center);

    float centerToCandidateDistance = nvcluster::length(centerToCandidateVector);

    float length = centerToCandidateDistance + candidate.radius + start.radius;
    //std::atomic_m
    if(std::isinf(length) || length > maxLength)
    {
      maxLength = length;
      result    = candidate;
    }
  }
  //NVLOD_PARALLEL_FOR_END;
  return result;
};

// Create a sphere that bounds all the input spheres
static inline nvclusterlod_Result makeBoundingSphere(std::span<const nvclusterlod_Sphere> spheres, nvclusterlod_Sphere& sphere)
{
  if(spheres.empty())
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_MAKE_BOUNDING_SPHERES_FROM_EMPTY_SET;
  }

  // Loosely based on Ritter's bounding sphere algorithm, extending to include
  // sphere radii. Not verified, but I can imagine it works.
  const nvclusterlod_Sphere& x = spheres[0];
  const nvclusterlod_Sphere  y = farthestSphere(spheres, x);
  const nvclusterlod_Sphere  z = farthestSphere(spheres, y);

  auto  yz   = fromAPI(z.center) - fromAPI(y.center);
  float dist = nvcluster::length(yz);
  yz *= 1.f / dist;

  nvcluster::vec3f resultCenter = fromAPI(y.center);
  float            resultRadius = (dist + y.radius + z.radius) * 0.5f;
  // TODO: I bet normalize could cancel down somehow to avoid the
  // singularity check
  if(dist > 1e-10f)
  {
    const float radiusDifference = resultRadius - y.radius;
    resultCenter += yz * radiusDifference;
  }
  sphere                = nvclusterlod_Sphere{toAPI(resultCenter), resultRadius};
  nvclusterlod_Sphere f = farthestSphere(spheres, sphere);

  nvcluster::vec3f sphereToFarthestVector   = fromAPI(f.center) - fromAPI(sphere.center);
  float            sphereToFarthestDistance = nvcluster::length(sphereToFarthestVector);

  sphere.radius = sphereToFarthestDistance + f.radius;
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());  // fixes a test failure. or * 1.0001?
  if(std::isnan(sphere.center.x) || std::isnan(sphere.center.y) || std::isnan(sphere.center.z) || std::isnan(sphere.radius))
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_BOUNDING_SPHERES;
  }


#ifndef NDEBUG
  for(size_t childIndex = 0; childIndex < spheres.size(); childIndex++)
  {
    const nvclusterlod_Sphere& child = spheres[childIndex];
    if(child.radius > sphere.radius)
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_BOUNDING_SPHERES;
    }
  }
#endif
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

// Compute the number of nodes required to store the LOD hierarchy for the given input
nvclusterlod_Result nvclusterlodGetHierarchyRequirements(nvclusterlod_Context,
                                                         const nvclusterlod_HierarchyInput* input,
                                                         nvclusterlod_HierarchyCounts*      counts)
{
  *counts = nvclusterlod_HierarchyCounts{
      .nodeCount = uint32_t(input->clusterCount + 1),
  };
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

nvclusterlod_Result nvclusterlodBuildHierarchy(nvclusterlod_Context               context,
                                               const nvclusterlod_HierarchyInput* inputPtr,
                                               nvclusterlod_HierarchyOutput*      output)
{
  const nvclusterlod_HierarchyInput& input = *inputPtr;
  // Build sets of generating groups that contributed clusters for decimation
  // into each group.
  nvclusterlod::GroupGeneratingGroups groupGeneratingGroups;

  std::span<const nvcluster_Range> groupClusterRanges(input.groupClusterRanges, input.groupCount);
  // FIXME: not sure about that count
  std::span<const uint32_t> clusterGeneratingGroups(input.clusterGeneratingGroups, input.clusterCount);

  nvclusterlod_Result result =
      nvclusterlod::generateGroupGeneratingGroups(groupClusterRanges, clusterGeneratingGroups, groupGeneratingGroups);
  if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }

  if(groupGeneratingGroups.ranges.size() != input.groupCount)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_GENERATING_GROUPS;
  }

  if(groupGeneratingGroups.ranges[0].offset != 0)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_GENERATING_GROUPS;
  }

  // Compute cumulative bounding spheres and quadric errors. Cumulative bounding
  // spheres avoid rendering overlapping geometry with a constant angular error
  // threshold at the cost of producing significantly oversized bounding
  // spheres.
  std::vector<nvclusterlod_Sphere>  groupCumulativeBoundingSpheres(input.groupCount);
  std::vector<float>                groupCumulativeQuadricErrors(input.groupCount, 0.0f);
  for(size_t lodLevel = 0; lodLevel < input.lodLevelCount; ++lodLevel)
  {
    const nvcluster_Range& lodGroupRange = input.lodLevelGroupRanges[lodLevel];
    for(uint32_t group = lodGroupRange.offset; group < lodGroupRange.offset + lodGroupRange.count; group++)
    {
      if(lodLevel == 0)
      {
        // Find the bounding sphere for each group
        result = makeBoundingSphere(std::span<const nvclusterlod_Sphere>(input.clusterBoundingSpheres
                                                                             + groupClusterRanges[group].offset,
                                                                         groupClusterRanges[group].count),
                                    groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
      }
      else
      {
        // Higher LOD bounding spheres just include the generating group
        // spheres. The current group will always be a subset, so no point
        // generating it.
        // TODO: only compute LOD0 clusterBoundingSpheres?
        std::vector<nvclusterlod_Sphere>  generatingSpheres;
        const nvcluster_Range&            generatingGroupRange = groupGeneratingGroups.ranges[group];
        generatingSpheres.reserve(generatingGroupRange.count);
        for(uint32_t indexInGeneratingGroups = generatingGroupRange.offset;
            indexInGeneratingGroups < generatingGroupRange.offset + generatingGroupRange.count; indexInGeneratingGroups++)
        {
          uint32_t generatingGroup = groupGeneratingGroups.groups[indexInGeneratingGroups];
          generatingSpheres.push_back(groupCumulativeBoundingSpheres[generatingGroup]);
        }
        result = makeBoundingSphere(generatingSpheres, groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
      }

      // Compute cumulative quadric error
      float                   maxGeneratingGroupQuadricError = 0.0f;
      const nvcluster_Range&  generatingGroupRange           = groupGeneratingGroups.ranges[group];
      for(uint32_t indexInGeneratingGroups = generatingGroupRange.offset;
          indexInGeneratingGroups < generatingGroupRange.offset + generatingGroupRange.count; indexInGeneratingGroups++)
      {
        uint32_t generatingGroup = groupGeneratingGroups.groups[indexInGeneratingGroups];
        maxGeneratingGroupQuadricError = std::max(maxGeneratingGroupQuadricError, groupCumulativeQuadricErrors[generatingGroup]);
      }
      groupCumulativeQuadricErrors[group] = maxGeneratingGroupQuadricError + input.groupQuadricErrors[group];
    }
  }

  // Write recursively propagated group data
  for(size_t i = 0; i < groupCumulativeBoundingSpheres.size(); i++)
  {
    output->groupCumulativeBoundingSpheres[i] = groupCumulativeBoundingSpheres[i];
  }


  for(size_t i = 0; i < groupCumulativeQuadricErrors.size(); i++)
  {
    output->groupCumulativeQuadricError[i] = groupCumulativeQuadricErrors[i];
  }

  // Allocate the initial root node, just so it is first
  size_t lodCount = input.lodLevelCount;
  if(lodCount >= NVCLUSTERLOD_NODE_MAX_CHILDREN)  // can fit all LODs into one root node.
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_LOD_OVERFLOW;
  }
  //nvlod::Node*             outNode  = output.nodes;
  nvclusterlod_HierarchyNode&             rootNode         = output->nodes[0];  // *outNode++;
  uint32_t                        currentNodeIndex = 1;
  std::vector<nvclusterlod_HierarchyNode> lodNodes;

  // Write the node hierarchy
  for(size_t lodIndex = 0; lodIndex < lodCount; ++lodIndex)
  {
    // Create leaf nodes for each group of clusters.
    std::vector<nvclusterlod_HierarchyNode> nodes;
    nodes.reserve(input.lodLevelGroupRanges[lodIndex].count);
    const nvcluster_Range& lodGroupRange = input.lodLevelGroupRanges[lodIndex];
    for(uint32_t groupIndex = lodGroupRange.offset; groupIndex < lodGroupRange.offset + lodGroupRange.count; groupIndex++)
    {
      if(input.groupClusterRanges[groupIndex].count > NVCLUSTERLOD_GROUP_MAX_CLUSTERS)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED;
      }
      nvclusterlod_LeafNodeClusterGroup clusterGroup{
          .isClusterGroup       = 1,
          .group                = groupIndex & U32_MASK(23),
          .clusterCountMinusOne = (input.groupClusterRanges[groupIndex].count - 1u) & U32_MASK(8),
      };
      if(uint32_t(clusterGroup.clusterCountMinusOne) + 1 != input.groupClusterRanges[groupIndex].count)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED;
      }
      nodes.push_back(nvclusterlod_HierarchyNode{
          .clusterGroup           = clusterGroup,
          .boundingSphere         = groupCumulativeBoundingSpheres[groupIndex],
          .maxClusterQuadricError = groupCumulativeQuadricErrors[groupIndex],
      });
    }

    // Build traversal hierarchy per-LOD
    // NOTE: could explore mixing nodes from different LODs near the top of the
    // tree to improve paralellism. Ideally the result could be N root nodes
    // rather than just one too.
    while(nodes.size() > 1)
    {
      nvcluster::ClusterStorage nodeClusters;
      nvcluster_Result          clusterResult =
          clusterNodesSpatially(context->clusterContext, nodes, NVCLUSTERLOD_NODE_MAX_CHILDREN, nodeClusters);
      if(clusterResult != nvcluster_Result::NVCLUSTER_SUCCESS)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTERING_FAILED;
      }
      std::vector<nvclusterlod_HierarchyNode> newNodes;
      newNodes.reserve(nodeClusters.clusterItemRanges.size());

      for(size_t rangeIndex = 0; rangeIndex < nodeClusters.clusterItemRanges.size(); rangeIndex++)
      {
        const nvcluster_Range& range = nodeClusters.clusterItemRanges[rangeIndex];
        std::span<uint32_t>    group = std::span<uint32_t>(nodeClusters.items).subspan(range.offset, range.count);
        if(group.empty() || group.size() > NVCLUSTERLOD_NODE_MAX_CHILDREN)
        {
          return nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED;
        }
        float                             maxClusterQuadricError = 0.0f;
        std::vector<nvclusterlod_Sphere>  boundingSpheres;
        boundingSpheres.reserve(group.size());
        for(uint32_t nodeIndex : group)
        {
          boundingSpheres.push_back(nodes[nodeIndex].boundingSphere);
          maxClusterQuadricError = std::max(maxClusterQuadricError, nodes[nodeIndex].maxClusterQuadricError);
        }
        nvclusterlod_InternalNodeChildren nodeRange{
            .isClusterGroup     = 0,
            .childOffset        = currentNodeIndex & U32_MASK(26),
            .childCountMinusOne = uint32_t(group.size() - 1) & U32_MASK(5),
        };
        nvclusterlod_Sphere boundingSphere;
        result = makeBoundingSphere(boundingSpheres, boundingSphere);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
        newNodes.push_back(nvclusterlod_HierarchyNode{
            .children               = nodeRange,
            .boundingSphere         = boundingSphere,
            .maxClusterQuadricError = maxClusterQuadricError,
        });

        for(const uint32_t& nodeIndex : group)
        {
          output->nodes[currentNodeIndex] = nodes[nodeIndex];
          currentNodeIndex++;
        }
      }
      std::swap(nodes, newNodes);
    }
    if(nodes.size() != 1)
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED;
    }
    // Always traverse lowest detail LOD by making the sphere radius huge
    if(lodIndex == lodCount - 1)
    {
      nodes[0].boundingSphere = {{0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::max()};
    }
    lodNodes.insert(lodNodes.end(), nodes.begin(), nodes.end());
  }

  // Link the per-LOD trees into a single root node
  // TODO: recursively add to support more than NodeRange::MaxChildren LOD levels
  {
    float maxClusterQuadricError = 0.0f;
    for(const nvclusterlod_HierarchyNode& node : lodNodes)
      maxClusterQuadricError = std::max(maxClusterQuadricError, node.maxClusterQuadricError);
    nvclusterlod_InternalNodeChildren nodeRange{
        .isClusterGroup     = 0,
        .childOffset        = currentNodeIndex & U32_MASK(26),
        .childCountMinusOne = uint32_t(lodNodes.size() - 1) & U32_MASK(5),
    };
    if(uint32_t(nodeRange.childCountMinusOne + 1) != uint32_t(lodNodes.size()))
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_NODE_OVERFLOW;
    }
    rootNode = nvclusterlod_HierarchyNode{
        .children               = nodeRange,
        .boundingSphere         = {{0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::max()},  // always include everything
        .maxClusterQuadricError = maxClusterQuadricError,
    };
    for(size_t nodeIndex = 0; nodeIndex < lodNodes.size(); nodeIndex++)
    {
      const nvclusterlod_HierarchyNode& node = lodNodes[nodeIndex];
      if(currentNodeIndex >= output->nodeCount)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_HIERARCHY_GENERATION_FAILED;
      }
      output->nodes[currentNodeIndex] = node;
      currentNodeIndex++;
    }
  }

  output->nodeCount = currentNodeIndex;
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}
