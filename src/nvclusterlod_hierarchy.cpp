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
#include <span>
#include <vector>

#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>

#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>

#include "nvclusterlod_context.hpp"
#include "nvclusterlod_parallel.hpp"

namespace nvclusterlod {

// Create a 32-bit mask with the lowest bitCount bits set to 1.
// bitCount must be less than 32.
#define U32_MASK(bitCount) ((1u << (bitCount)) - 1u)

// From the set of input nodes, cluster them according to their spatial location so each cluster contains at most maxClusterItems
static nvcluster::Result clusterNodesSpatially(nvcluster::Context                  context,
                                               std::span<const nvclusterlod::Node> nodes,
                                               uint32_t                            maxClusterItems,
                                               nvcluster::ClusterStorage&          clusters)
{
  // For each node, compute its axis-aligned bounding box and centroid location
  std::vector<nvcluster::AABB> triangleClusterAabbs(nodes.size());
  std::vector<float>           triangleClusterCentroids(nodes.size() * 3);

  NVLOD_PARALLEL_FOR_BEGIN(nodeIndex, nodes.size(), 2048)
  {
    const nvclusterlod::Node& node      = nodes[nodeIndex];
    float                     center[3] = {node.boundingSphere.x, node.boundingSphere.y, node.boundingSphere.z};

    nvcluster::AABB& aabb = triangleClusterAabbs[nodeIndex];
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
  nvcluster::SpatialElements clusterBounds{};
  clusterBounds.boundingBoxes = triangleClusterAabbs.data();

  clusterBounds.centroids    = triangleClusterCentroids.data();
  clusterBounds.elementCount = uint32_t(triangleClusterAabbs.size());

  nvcluster::Input clusterGroupInput{};
  clusterGroupInput.config = {
      .minClusterSize = maxClusterItems,
      .maxClusterSize = maxClusterItems,
  };

  clusterGroupInput.spatialElements = &clusterBounds;
  clusterGroupInput.graph           = nullptr;


  nvcluster::Result result = nvcluster::generateClusters(context, clusterGroupInput, clusters);
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
inline nvclusterlod::Sphere farthestSphere(std::span<const nvclusterlod::Sphere> spheres, const nvclusterlod::Sphere& start)
{
  nvclusterlod::Sphere result    = start;
  float                maxLength = 0.0f;

  // FIXME: todo for parallelism
  //SphereDist sd{.u64 = 0ull};


  // FIXME: parallelize?
  //for(const nvlod::Sphere& candidate : spheres)
  //NVLOD_PARALLEL_FOR_BEGIN(sphereIndex, spheres.size())
  for(size_t sphereIndex = 0; sphereIndex < spheres.size(); sphereIndex++)
  {
    const Sphere& candidate                  = spheres[sphereIndex];
    float         centerToCandidateVector[3] = {candidate.x - start.x, candidate.y - start.y, candidate.z - start.z};

    float centerToCandidateDistance = std::sqrt(centerToCandidateVector[0] * centerToCandidateVector[0]
                                                + centerToCandidateVector[1] * centerToCandidateVector[1]
                                                + centerToCandidateVector[2] * centerToCandidateVector[2]);

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
static inline nvclusterlod::Result makeBoundingSphere(std::span<const nvclusterlod::Sphere> spheres, nvclusterlod::Sphere& sphere)
{
  if(spheres.empty())
  {
    return {};
  }

  // Loosely based on Ritter's bounding sphere algorithm, extending to include
  // sphere radii. Not verified, but I can imagine it works.
  const nvclusterlod::Sphere& x = spheres[0];
  nvclusterlod::Sphere        y = farthestSphere(spheres, x);
  nvclusterlod::Sphere        z = farthestSphere(spheres, y);

  float yz[3]   = {z.x - y.x, z.y - y.y, z.z - y.z};
  float dist    = std::sqrt(yz[0] * yz[0] + yz[1] * yz[1] + yz[2] * yz[2]);
  float invDist = 1.f / dist;
  yz[0] *= invDist;
  yz[1] *= invDist;
  yz[2] *= invDist;

  float resultRadius = (dist + y.radius + z.radius) * 0.5f;
  sphere             = nvclusterlod::Sphere{y.x, y.y, y.z, resultRadius};
  // TODO: I bet normalize could cancel down somehow to avoid the
  // singularity check
  if(dist > 1e-10f)
  {
    const float radiusDifference = resultRadius - y.radius;
    sphere.x += yz[0] * radiusDifference;
    sphere.y += yz[1] * radiusDifference;
    sphere.z += yz[2] * radiusDifference;
  }
  nvclusterlod::Sphere f = farthestSphere(spheres, sphere);

  float sphereToFarthestVector[3] = {f.x - sphere.x, f.y - sphere.y, f.z - sphere.z};
  float sphereToFarthestDistance =
      std::sqrt(sphereToFarthestVector[0] * sphereToFarthestVector[0] + sphereToFarthestVector[1] * sphereToFarthestVector[1]
                + sphereToFarthestVector[2] * sphereToFarthestVector[2]);

  sphere.radius = sphereToFarthestDistance + f.radius;
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());  // fixes a test failure. or * 1.0001?
  if(std::isnan(sphere.x) || std::isnan(sphere.y) || std::isnan(sphere.z) || std::isnan(sphere.radius))
  {
    return nvclusterlod::Result::ERROR_INCONSISTENT_BOUNDING_SPHERES;
  }


#ifndef NDEBUG
  for(size_t childIndex = 0; childIndex < spheres.size(); childIndex++)
  {
    const Sphere& child = spheres[childIndex];
    if(child.radius > sphere.radius)
    {
      return nvclusterlod::Result::ERROR_INCONSISTENT_BOUNDING_SPHERES;
    }
  }
#endif
  return nvclusterlod::Result::SUCCESS;
}
}  // namespace nvclusterlod

// Compute the number of nodes required to store the LOD hierarchy for the given input
nvclusterlod::HierarchyRequirements nvclusterlodHierarchyGetRequirements(nvclusterlod::Context /*context*/,
                                                                         const nvclusterlod::HierarchyGetRequirementsInfo* info)
{
  nvclusterlod::HierarchyRequirements result{};
  result.maxNodeCount = uint32_t(info->input->clusterCount + 1);
  return result;
}


nvclusterlod::Result nvclusterlodCreateHierarchy(nvclusterlod::Context                    context,
                                                 const nvclusterlod::HierarchyCreateInfo* info,
                                                 nvclusterlod::HierarchyOutput*           output)
{
  const nvclusterlod::HierarchyInput& input = *info->input;
  // Build sets of generating groups that contributed clusters for decimation
  // into each group.
  nvclusterlod::GroupGeneratingGroups groupGeneratingGroups;

  std::span<const nvcluster::Range> groupClusterRanges(input.groupClusterRanges, input.groupCount);
  // FIXME: not sure about that count
  std::span<const uint32_t> clusterGeneratingGroups(input.clusterGeneratingGroups, input.clusterCount);

  nvclusterlod::Result result =
      nvclusterlod::generateGroupGeneratingGroups(groupClusterRanges, clusterGeneratingGroups, groupGeneratingGroups);
  if(result != nvclusterlod::Result::SUCCESS)
  {
    return result;
  }

  if(groupGeneratingGroups.ranges.size() != input.groupCount)
  {
    return nvclusterlod::Result::ERROR_INCONSISTENT_GENERATING_GROUPS;
  }

  if(groupGeneratingGroups.ranges[0].offset != 0)
  {
    return nvclusterlod::Result::ERROR_INCONSISTENT_GENERATING_GROUPS;
  }

  // Compute cumulative bounding spheres and quadric errors. Cumulative bounding
  // spheres avoid rendering overlapping geometry with a constant angular error
  // threshold at the cost of producing significantly oversized bounding
  // spheres.
  std::vector<nvclusterlod::Sphere> groupCumulativeBoundingSpheres(input.groupCount);
  std::vector<float>                groupCumulativeQuadricErrors(input.groupCount, 0.0f);
  for(size_t lodLevel = 0; lodLevel < input.lodLevelCount; ++lodLevel)
  {
    const nvcluster::Range& lodGroupRange = input.lodLevelGroupRanges[lodLevel];
    for(uint32_t group = lodGroupRange.offset; group < lodGroupRange.offset + lodGroupRange.count; group++)
    {
      if(lodLevel == 0)
      {
        // Find the bounding sphere for each group
        result = makeBoundingSphere(std::span<const nvclusterlod::Sphere>(input.clusterBoundingSpheres
                                                                              + groupClusterRanges[group].offset,
                                                                          groupClusterRanges[group].count),
                                    groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod::Result::SUCCESS)
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
        std::vector<nvclusterlod::Sphere> generatingSpheres;
        const nvcluster::Range&           generatingGroupRange = groupGeneratingGroups.ranges[group];
        generatingSpheres.reserve(generatingGroupRange.count);
        for(uint32_t indexInGeneratingGroups = generatingGroupRange.offset;
            indexInGeneratingGroups < generatingGroupRange.offset + generatingGroupRange.count; indexInGeneratingGroups++)
        {
          uint32_t generatingGroup = groupGeneratingGroups.groups[indexInGeneratingGroups];
          generatingSpheres.push_back(groupCumulativeBoundingSpheres[generatingGroup]);
        }
        result = makeBoundingSphere(generatingSpheres, groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod::Result::SUCCESS)
        {
          return result;
        }
      }

      // Compute cumulative quadric error
      float                   maxGeneratingGroupQuadricError = 0.0f;
      const nvcluster::Range& generatingGroupRange           = groupGeneratingGroups.ranges[group];
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
  if(lodCount >= nvclusterlod::InteriorNode::NODE_RANGE_MAX_SIZE)  // can fit all LODs into one root node.
  {
    return nvclusterlod::Result::ERROR_LOD_OVERFLOW;
  }
  //nvlod::Node*             outNode  = output.nodes;
  nvclusterlod::Node&             rootNode         = output->nodes[0];  // *outNode++;
  uint32_t                        currentNodeIndex = 1;
  std::vector<nvclusterlod::Node> lodNodes;

  // Write the node hierarchy
  for(size_t lodIndex = 0; lodIndex < lodCount; ++lodIndex)
  {
    // Create leaf nodes for each group of clusters.
    std::vector<nvclusterlod::Node> nodes;
    nodes.reserve(input.lodLevelGroupRanges[lodIndex].count);
    const nvcluster::Range& lodGroupRange = input.lodLevelGroupRanges[lodIndex];
    for(uint32_t groupIndex = lodGroupRange.offset; groupIndex < lodGroupRange.offset + lodGroupRange.count; groupIndex++)
    {
      if(input.groupClusterRanges[groupIndex].count > nvclusterlod::LeafNode::CLUSTER_RANGE_MAX_SIZE)
      {
        return nvclusterlod::Result::ERROR_HIERARCHY_GENERATION_FAILED;
      }
      nvclusterlod::LeafNode clusterRange{
          .isLeafNode           = 1,
          .group                = groupIndex & U32_MASK(23),
          .clusterCountMinusOne = (input.groupClusterRanges[groupIndex].count - 1u) & U32_MASK(8),
      };
      if(uint32_t(clusterRange.clusterCountMinusOne) + 1 != input.groupClusterRanges[groupIndex].count)
      {
        return nvclusterlod::Result::ERROR_HIERARCHY_GENERATION_FAILED;
      }
      nodes.push_back(nvclusterlod::Node{
          .clusters               = clusterRange,
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
      nvcluster::Result         clusterResult =
          clusterNodesSpatially(context->clusterContext, nodes, nvclusterlod::InteriorNode::NODE_RANGE_MAX_SIZE, nodeClusters);
      if(clusterResult != nvcluster::Result::SUCCESS)
      {
        return nvclusterlod::Result::ERROR_CLUSTERING_FAILED;
      }
      std::vector<nvclusterlod::Node> newNodes;
      newNodes.reserve(nodeClusters.clusterRanges.size());

      for(size_t rangeIndex = 0; rangeIndex < nodeClusters.clusterRanges.size(); rangeIndex++)
      {
        const nvcluster::Range& range = nodeClusters.clusterRanges[rangeIndex];
        std::span<uint32_t> group = std::span<uint32_t>(nodeClusters.clusterItems).subspan(range.offset, range.count);
        if(group.empty() || group.size() > nvclusterlod::InteriorNode::NODE_RANGE_MAX_SIZE)
        {
          return nvclusterlod::Result::ERROR_HIERARCHY_GENERATION_FAILED;
        }
        float                             maxClusterQuadricError = 0.0f;
        std::vector<nvclusterlod::Sphere> boundingSpheres;
        boundingSpheres.reserve(group.size());
        for(uint32_t nodeIndex : group)
        {
          boundingSpheres.push_back(nodes[nodeIndex].boundingSphere);
          maxClusterQuadricError = std::max(maxClusterQuadricError, nodes[nodeIndex].maxClusterQuadricError);
        }
        nvclusterlod::InteriorNode nodeRange{
            .isLeafNode         = 0,
            .childOffset        = currentNodeIndex & U32_MASK(26),
            .childCountMinusOne = uint32_t(group.size() - 1) & U32_MASK(5),
        };
        nvclusterlod::Sphere boundingSphere;
        result = makeBoundingSphere(boundingSpheres, boundingSphere);
        if(result != nvclusterlod::Result::SUCCESS)
        {
          return result;
        }
        newNodes.push_back(nvclusterlod::Node{
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
      return nvclusterlod::Result::ERROR_HIERARCHY_GENERATION_FAILED;
    }
    // Always traverse lowest detail LOD by making the sphere radius huge
    if(lodIndex == lodCount - 1)
    {
      nodes[0].boundingSphere = {0.0f, 0.0f, 0.0f, std::numeric_limits<float>::max()};
    }
    lodNodes.insert(lodNodes.end(), nodes.begin(), nodes.end());
  }

  // Link the per-LOD trees into a single root node
  // TODO: recursively add to support more than NodeRange::MaxChildren LOD levels
  {
    float maxClusterQuadricError = 0.0f;
    for(const nvclusterlod::Node& node : lodNodes)
      maxClusterQuadricError = std::max(maxClusterQuadricError, node.maxClusterQuadricError);
    nvclusterlod::InteriorNode nodeRange{
        .isLeafNode         = 0,
        .childOffset        = currentNodeIndex & U32_MASK(26),
        .childCountMinusOne = uint32_t(lodNodes.size() - 1) & U32_MASK(5),
    };
    if(uint32_t(nodeRange.childCountMinusOne + 1) != uint32_t(lodNodes.size()))
    {
      return nvclusterlod::Result::ERROR_NODE_OVERFLOW;
    }
    rootNode = nvclusterlod::Node{
        .children               = nodeRange,
        .boundingSphere         = {0.0f, 0.0f, 0.0f, std::numeric_limits<float>::max()},  // always include everything
        .maxClusterQuadricError = maxClusterQuadricError,
    };
    for(size_t nodeIndex = 0; nodeIndex < lodNodes.size(); nodeIndex++)
    {
      const nvclusterlod::Node& node = lodNodes[nodeIndex];
      if(currentNodeIndex >= output->nodeCount)
      {
        return nvclusterlod::Result::ERROR_HIERARCHY_GENERATION_FAILED;
      }
      output->nodes[currentNodeIndex] = node;
      currentNodeIndex++;
    }
  }

  output->nodeCount = currentNodeIndex;
  return nvclusterlod::Result::SUCCESS;
}


//}  // namespace nvlod_internal
