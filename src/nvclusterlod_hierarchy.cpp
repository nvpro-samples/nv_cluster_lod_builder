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

#include <bit>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <nvcluster/util/parallel.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod_context.hpp>
#include <nvclusterlod_cpp.hpp>
#include <span>
#include <vector>

// Create a 32-bit mask with the lowest bitCount bits set to 1.
// bitCount must be less than 32.
#define U32_MASK(bitCount) ((1u << (bitCount)) - 1u)

namespace nvclusterlod {

// From the set of input nodes, cluster them according to their spatial location so each cluster contains at most maxClusterItems
template <bool Parallelize>
static nvcluster_Result clusterNodesSpatially(nvcluster_Context                           context,
                                              std::span<const nvclusterlod_HierarchyNode> nodes,
                                              uint32_t                                    maxClusterItems,
                                              nvcluster::ClusterStorage&                  clusters)
{
  // For each node, compute its axis-aligned bounding box and centroid location
  std::vector<nvcluster::AABB>  triangleClusterAabbs(nodes.size());
  std::vector<nvcluster::vec3f> triangleClusterCentroids(nodes.size());

  parallel_batches<Parallelize, 2048>(nodes.size(), [&](uint64_t nodeIndex) {
    const nvclusterlod_HierarchyNode& node           = nodes[nodeIndex];
    auto                              boundingSphere = std::bit_cast<nvclusterlod::Sphere>(node.boundingSphere);
    triangleClusterAabbs[nodeIndex]                  = {boundingSphere.center - boundingSphere.radius,
                                                        boundingSphere.center + boundingSphere.radius};
    triangleClusterCentroids[nodeIndex]              = boundingSphere.center;
  });

  // Call the clusterizer to group the nodes
  nvcluster_Input clusterBounds{
      .itemBoundingBoxes = reinterpret_cast<const nvcluster_AABB*>(triangleClusterAabbs.data()),
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

// Find the sphere within spheres that lies farthest from the target sphere,
// accounting for the radii of the spheres
inline const nvclusterlod::Sphere farthestSphere(std::span<const nvclusterlod::Sphere> spheres, const nvclusterlod::Sphere& target)
{
  const nvclusterlod::Sphere* result  = &target;
  float                       maxDist = 0.0f;
  for(size_t sphereIndex = 0; sphereIndex < spheres.size(); sphereIndex++)
  {
    const nvclusterlod::Sphere& candidate = spheres[sphereIndex];
    float dist = nvcluster::length(candidate.center - target.center) + candidate.radius + target.radius;
    if(std::isinf(dist) || dist > maxDist)
    {
      maxDist = dist;
      result  = &candidate;
    }
  }
  return *result;
};

// Create a sphere that bounds all the input spheres
static inline nvclusterlod_Result makeBoundingSphere(std::span<const nvclusterlod::Sphere> spheres, nvclusterlod::Sphere& sphere)
{
  if(spheres.empty())
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_MAKE_BOUNDING_SPHERES_FROM_EMPTY_SET;
  }

  // Loosely based on Ritter's bounding sphere algorithm, extending to include
  // sphere radii. Not verified, but I can imagine it works.
  const nvclusterlod::Sphere& x = spheres[0];
  const nvclusterlod::Sphere  y = farthestSphere(spheres, x);
  const nvclusterlod::Sphere  z = farthestSphere(spheres, y);

  // Make a sphere containing y and z
  auto  yz   = z.center - y.center;
  float dist = nvcluster::length(yz);
  sphere     = {y.center, (dist + y.radius + z.radius) * 0.5f};
  // TODO: I bet normalize could cancel down somehow to avoid the
  // singularity check
  if(dist > 1e-10f)
    sphere.center += yz * (sphere.radius - y.radius) / dist;

  // Grow the sphere to include the farthest sphere
  const nvclusterlod::Sphere f = farthestSphere(spheres, sphere);
  sphere.radius                = nvcluster::length(f.center - sphere.center) + f.radius;
  sphere.radius                = std::nextafter(sphere.radius, std::numeric_limits<float>::max());
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());  // fixes a test failure. or * 1.0001?
  sphere.radius = std::nextafter(sphere.radius, std::numeric_limits<float>::max());
  if(std::isnan(sphere.center[0]) || std::isnan(sphere.center[1]) || std::isnan(sphere.center[2]) || std::isnan(sphere.radius))
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_PRODUCED_NAN_BOUNDING_SPHERES;
  }

#ifndef NDEBUG
  for(size_t childIndex = 0; childIndex < spheres.size(); childIndex++)
  {
    assert(spheres[childIndex].radius <= sphere.radius);
    assert(isInside(spheres[childIndex], sphere));
  }
#endif
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

template <bool Parallelize>
nvclusterlod_Result buildHierarchy(nvclusterlod_Context context, const HierarchyInput& input, HierarchyOutput& output)
{
  // Build sets of generating groups that contributed clusters for decimation
  // into each group.
  nvclusterlod::GroupGeneratingGroups groupGeneratingGroups;

  std::span           groupClusterRangesCAPI(reinterpret_cast<const nvcluster_Range*>(input.groupClusterRanges.data()),
                                             input.groupClusterRanges.size());
  nvclusterlod_Result result =
      nvclusterlod::generateGroupGeneratingGroups(groupClusterRangesCAPI, input.clusterGeneratingGroups, groupGeneratingGroups);
  if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return result;
  }

  // Compute cumulative bounding spheres and quadric errors. Cumulative bounding
  // spheres avoid rendering overlapping geometry with a constant angular error
  // threshold at the cost of producing significantly oversized bounding
  // spheres.
  for(size_t lodLevel = 0; lodLevel < input.lodLevelGroupRanges.size(); ++lodLevel)
  {
    const nvcluster::Range& lodGroupRange = input.lodLevelGroupRanges[lodLevel];
    for(uint32_t group = lodGroupRange.offset; group < lodGroupRange.end(); group++)
    {
      if(lodLevel == 0)
      {
        // Find the bounding sphere for each group
        result = makeBoundingSphere(input.clusterBoundingSpheres.subspan(input.groupClusterRanges[group].offset,
                                                                         input.groupClusterRanges[group].count),
                                    output.groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
      }
      else
      {
        // Higher LOD bounding spheres just include the generating group
        // spheres. The current group will seemingly always be a subset.
        // However, since the bounding sphere algorithm isn't perfectly tight
        // it's possible that a cluster bounding sphere may be outside the one
        // computed here. This isn't important for LOD but can be surprising if
        // validated.
        // TODO: only compute LOD0 clusterBoundingSpheres from triangles?
        std::vector<nvclusterlod::Sphere> generatingSpheres;
        const nvcluster_Range&            generatingGroupRange = groupGeneratingGroups.ranges[group];
        generatingSpheres.reserve(generatingGroupRange.count);
        for(uint32_t indexInGeneratingGroups = generatingGroupRange.offset;
            indexInGeneratingGroups < generatingGroupRange.offset + generatingGroupRange.count; indexInGeneratingGroups++)
        {
          uint32_t generatingGroup = groupGeneratingGroups.groups[indexInGeneratingGroups];
          generatingSpheres.push_back(output.groupCumulativeBoundingSpheres[generatingGroup]);
        }
        result = makeBoundingSphere(generatingSpheres, output.groupCumulativeBoundingSpheres[group]);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
      }

      // Compute cumulative quadric error
      float                  maxGeneratingGroupQuadricError = 0.0f;
      const nvcluster_Range& generatingGroupRange           = groupGeneratingGroups.ranges[group];
      for(uint32_t indexInGeneratingGroups = generatingGroupRange.offset;
          indexInGeneratingGroups < generatingGroupRange.offset + generatingGroupRange.count; indexInGeneratingGroups++)
      {
        uint32_t generatingGroup = groupGeneratingGroups.groups[indexInGeneratingGroups];
        maxGeneratingGroupQuadricError =
            std::max(maxGeneratingGroupQuadricError, output.groupCumulativeQuadricError[generatingGroup]);
      }
      output.groupCumulativeQuadricError[group] = maxGeneratingGroupQuadricError + input.groupQuadricErrors[group];
    }
  }

  // Allocate the initial root node, just so it is first
  size_t lodCount = input.lodLevelGroupRanges.size();
  if(lodCount == 0)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_EMPTY_LOD_LEVELS;
  }
  if(lodCount >= NVCLUSTERLOD_NODE_MAX_CHILDREN)  // can fit all LODs into one root node.
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_LOD_LEVELS_OVERFLOW;
  }

  // The very first node is the root node.
  nvclusterlod_HierarchyNode& rootNode = output.nodes.allocate();

  // The root node children are next. Root children are a per-LOD spatial
  // hierarchy. They are combined for convenience. Note that lodNodes are
  // allocated here, but written after we have written their descendents.
  uint32_t                               lodNodesGlobalOffset = output.nodes.allocatedCount();
  OutputSpan<nvclusterlod_HierarchyNode> lodNodes             = output.nodes.allocate(uint32_t(lodCount));

  // Write the spatial hierarchy for each LOD level
  for(size_t lodIndex = 0; lodIndex < lodCount; ++lodIndex)
  {
    // Create leaf nodes for each group of clusters.
    std::vector<nvclusterlod_HierarchyNode> nodes;
    nodes.reserve(input.lodLevelGroupRanges[lodIndex].count);
    const nvcluster::Range& lodGroupRange = input.lodLevelGroupRanges[lodIndex];
    for(uint32_t groupIndex = lodGroupRange.offset; groupIndex < lodGroupRange.end(); groupIndex++)
    {
      if(input.groupClusterRanges[groupIndex].count > NVCLUSTERLOD_GROUP_MAX_CLUSTERS)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_GROUP_CLUSTER_COUNT_OVERFLOW;
      }
      nvclusterlod_LeafNodeClusterGroup clusterGroup{
          .isClusterGroup       = 1,
          .group                = groupIndex & U32_MASK(23),
          .clusterCountMinusOne = (input.groupClusterRanges[groupIndex].count - 1u) & U32_MASK(8),
      };
      assert(uint32_t(clusterGroup.clusterCountMinusOne) + 1 == input.groupClusterRanges[groupIndex].count);
      nodes.push_back(nvclusterlod_HierarchyNode{
          .clusterGroup           = clusterGroup,
          .boundingSphere         = output.groupCumulativeBoundingSpheres[groupIndex],
          .maxClusterQuadricError = output.groupCumulativeQuadricError[groupIndex],
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
          clusterNodesSpatially<Parallelize>(context->clusterContext, nodes, NVCLUSTERLOD_NODE_MAX_CHILDREN, nodeClusters);
      if(clusterResult != nvcluster_Result::NVCLUSTER_SUCCESS)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTERING_NODES_FAILED;
      }
      std::vector<nvclusterlod_HierarchyNode> newNodes;
      newNodes.reserve(nodeClusters.clusterItemRanges.size());

      for(size_t rangeIndex = 0; rangeIndex < nodeClusters.clusterItemRanges.size(); rangeIndex++)
      {
        const nvcluster_Range& range = nodeClusters.clusterItemRanges[rangeIndex];
        std::span<uint32_t>    group = std::span<uint32_t>(nodeClusters.items).subspan(range.offset, range.count);
        if(group.empty() || group.size() > NVCLUSTERLOD_NODE_MAX_CHILDREN)
        {
          return nvclusterlod_Result::NVCLUSTERLOD_ERROR_NODE_CHILD_COUNT_OVERFLOW;
        }
        float                             maxClusterQuadricError = 0.0f;
        std::vector<nvclusterlod::Sphere> boundingSpheres;
        boundingSpheres.reserve(group.size());
        for(uint32_t nodeIndex : group)
        {
          boundingSpheres.push_back(std::bit_cast<nvclusterlod::Sphere>(nodes[nodeIndex].boundingSphere));
          maxClusterQuadricError = std::max(maxClusterQuadricError, nodes[nodeIndex].maxClusterQuadricError);
        }
        nvclusterlod_InternalNodeChildren nodeRange{
            .isClusterGroup     = 0,
            .childOffset        = output.nodes.allocatedCount() & U32_MASK(26),
            .childCountMinusOne = uint32_t(group.size() - 1) & U32_MASK(5),
        };
        nvclusterlod::Sphere boundingSphere;
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
          output.nodes.append(nodes[nodeIndex]);
        }
      }
      std::swap(nodes, newNodes);
    }
    assert(nodes.size() == 1);

    // Always traverse lowest detail LOD by making the sphere radius huge. The
    // application may want this information, but can read it from the last
    // groupCumulativeBoundingSpheres instead.
    if(lodIndex == lodCount - 1)
    {
      nodes[0].boundingSphere = {{0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::max()};
    }
    lodNodes.append(nodes);
  }
  assert(lodNodes.allocatedCount() == lodNodes.capacity());

  // Link the per-LOD trees into a single root node
  // TODO: would need to combine recursively to support more than
  // NodeRange::MaxChildren LOD levels
  {
    float maxClusterQuadricError = 0.0f;
    for(const nvclusterlod_HierarchyNode& node : lodNodes.allocated())
      maxClusterQuadricError = std::max(maxClusterQuadricError, node.maxClusterQuadricError);
    nvclusterlod_InternalNodeChildren nodeRange{
        .isClusterGroup     = 0,
        .childOffset        = lodNodesGlobalOffset & U32_MASK(26),
        .childCountMinusOne = (lodNodes.allocatedCount() - 1) & U32_MASK(5),
    };
    if(uint32_t(nodeRange.childCountMinusOne + 1) != lodNodes.allocatedCount())
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_NODES_OVERFLOW;
    }
    rootNode = nvclusterlod_HierarchyNode{
        .children               = nodeRange,
        .boundingSphere         = {{0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::max()},  // always include everything
        .maxClusterQuadricError = maxClusterQuadricError,
    };
  }
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

}  // namespace nvclusterlod

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

// C API entrypoint
nvclusterlod_Result nvclusterlodBuildHierarchy(nvclusterlod_Context               context,
                                               const nvclusterlod_HierarchyInput* input,
                                               nvclusterlod_HierarchyOutput*      output)
{
  nvclusterlod::HierarchyOutput outputAllocator(*output, input->groupCount);
#if !defined(NVCLUSTERLOD_MULTITHREADED) || NVCLUSTERLOD_MULTITHREADED
  auto buildHierarchy = context->parallelize ? nvclusterlod::buildHierarchy<true> : nvclusterlod::buildHierarchy<false>;
#else
  auto buildHierarchy = nvclusterlod::buildHierarchy<false>;
#endif
  if(nvclusterlod_Result r = buildHierarchy(context, nvclusterlod::HierarchyInput::fromCAPI(*input), outputAllocator);
     r != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return r;
  }
  outputAllocator.writeCounts(*output);
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}
