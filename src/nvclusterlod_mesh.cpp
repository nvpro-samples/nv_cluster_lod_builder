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

#include <array>
#include <array_view.hpp>
#include <cstdint>
#include <execution>
#include <meshoptimizer.h>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <nvclusterlod/nvclusterlod_common.h>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <nvclusterlod_context.hpp>
#include <nvclusterlod_parallel.hpp>
#include <nvclusterlod_util.hpp>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static constexpr uint32_t NVLOD_MINIMAL_ADJACENCY_SIZE          = 5;
static constexpr uint32_t NVLOD_LOCKED_VERTEX_WEIGHT_MULTIPLIER = 10;
static constexpr float    NVLOD_VERTEX_WEIGHT_MULTIPLIER        = 10.f;

#define PRINT_OPS 0
//#define PRINT_PERF 1
#define WRITE_DECIMATION_FAILURE_OBJS 0

#if PRINT_OPS
#define DEBUG_PRINT(f_, ...) fprintf(stderr, (f_), ##__VA_ARGS__)
#else
#define DEBUG_PRINT(f_, ...)
#endif

#if WRITE_DECIMATION_FAILURE_OBJS
#include <fstream>
#include <ostream>
#endif

// Scoped profiler for quick and coarse results
// https://stackoverflow.com/questions/31391914/timing-in-an-elegant-way-in-c
#if PRINT_PERF
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#endif


class Stopwatch
{
#if PRINT_PERF
public:
  Stopwatch(std::string name)
      : m_name(std::move(name))
      , m_beg(std::chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " ms\n";
  }

private:
  std::string                                                 m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
#else
public:
  template <class T>
  Stopwatch([[maybe_unused]] T&& t)
  {
  }
#endif
};

namespace nvclusterlod {

using vec3  = nvcluster::vec3f;
using vec3u = nvcluster::vec3u;

static inline nvcluster_AABB emptyAABB()
{
  nvcluster_AABB res;
  res.bboxMin[0] = std::numeric_limits<float>::max();
  res.bboxMin[1] = std::numeric_limits<float>::max();
  res.bboxMin[2] = std::numeric_limits<float>::max();
  res.bboxMax[0] = std::numeric_limits<float>::lowest();
  res.bboxMax[1] = std::numeric_limits<float>::lowest();
  res.bboxMax[2] = std::numeric_limits<float>::lowest();
  return res;
}

static inline void addAABB(nvcluster_AABB& aabb, const nvcluster_AABB& added)
{
  for(uint32_t i = 0; i < 3; i++)
  {
    aabb.bboxMin[i] = std::min(aabb.bboxMin[i], added.bboxMin[i]);
    aabb.bboxMax[i] = std::max(aabb.bboxMax[i], added.bboxMax[i]);
  }
}


// Clustered triangles for all groups, hence SegmentedClustering. Each segment
// is a group from the previous LOD iteration. Within each segment triangles are
// re-clustered.
struct TriangleClusters
{
  nvcluster::SegmentedClusterStorage clustering;

  // Bounding boxes for each cluster
  std::vector<nvcluster_AABB> clusterAabbs;

  // Triangles are clustered from ranges of input geometry. The initial
  // clustering has one range - the whole mesh. Subsequent ranges come from
  // decimated cluster groups of the previous level. The generating group is
  // implicitly generatingGroupOffset plus the segment index.
  uint32_t generatingGroupOffset;

  uint32_t maxClusterItems;
};

// Shared vertex counts between triangle clusters. This is used to compute
// adjacency weights for grouping triangle clusters. Connections are symmetric,
// and the data is duplicated (i.e. clusterAdjacencyConnections[0][1] will have
// the same counts as clusterAdjacencyConnections[1][0]).
struct AdjacencyVertexCount
{
  uint16_t vertexCount = 0;
  uint16_t lockedCount = 0;
};

typedef std::vector<std::unordered_map<uint32_t, AdjacencyVertexCount>> ClusterAdjacency;

// Groups of triangle clusters
struct ClusterGroups
{
  nvcluster::ClusterStorage groups{};
  std::vector<uint32_t>     groupTriangleCounts;  // useful byproduct
  uint32_t                  totalTriangleCount{0};
  uint32_t                  globalGroupOffset{0};
};

// Groups of triangles. This is both for algorithm input and LOD recursion.
struct DecimatedClusterGroups
{
  // Ranges of triangles. Initially there is just one range containing the input
  // mesh. In subsequent iterations each ranges is a group of decimated
  // triangles. Clusters of triangles are formed within each range.
  std::vector<nvcluster_Range> groupTriangleRanges;

  // Triangle indices and vertices. Triangles are grouped by
  // groupTriangleRanges. Vertices always point to the input mesh vertices.
  nvclusterlod_MeshInput mesh;

  // Storage for decimated triangles from the previous pass. Note that triangles
  // are written to the output in clusters, which are formed from these at the
  // start of each iteration.
  std::vector<nvcluster::vec3u> decimatedTriangleStorage;

  // Per-group quadric errors from decimation
  std::vector<float> groupQuadricErrors;

  // Added to the groupTriangleRanges index for a global group index. This is
  // needed to write clusterGeneratingGroups.
  uint32_t baseClusterGroupIndex{0u};

  // Boolean list of locked vertices from the previous pass. Used to encourage
  // connecting clusters with shared locked vertices by increasing adjacency
  // weights.
  std::vector<uint8_t> globalLockedVertices;
};


struct OutputWritePositions
{
  uint32_t clusterTriangleRange{0u};
  uint32_t clusterTriangleVertex{0u};
  uint32_t clusterParentGroup{0u};
  uint32_t clusterBoundingSphere{0u};
  uint32_t groupQuadricError{0u};
  uint32_t groupCluster{0u};
  uint32_t lodLevelGroup{0u};
};

// Find the vertex in the mesh that is the farthest from the start point.
inline void farthestPoint(const nvclusterlod_MeshInput& mesh, const nvcluster::vec3f& start, nvcluster::vec3f& farthest)
{
  const nvcluster_Vec3f* result = nullptr;

  float maxLengthSq = 0.0f;

  // Iterate over triangles, paying the cost of visiting duplicate vertices so
  // that unused vertices are not included.
  for(uint32_t triangleIndex = 0; triangleIndex < mesh.triangleCount; triangleIndex++)
  {
    nvcluster::vec3u triangle = fromAPI(mesh.triangleVertices[triangleIndex]);
    for(size_t i = 0; i < 3; ++i)
    {
      const nvcluster_Vec3f* candidatePtr = &ArrayView(mesh.vertexPositions, mesh.vertexCount, mesh.vertexStride)[triangle[i]];
      float lengthSq = nvcluster::length_squared(fromAPI(*candidatePtr) - start);
      if(lengthSq > maxLengthSq)
      {
        maxLengthSq = lengthSq;
        result      = candidatePtr;
      }
    }
  }

  if(result != nullptr)
  {
    farthest = fromAPI(*result);
  }
};

// Ritter's bounding sphere algorithm
// https://en.wikipedia.org/wiki/Bounding_sphere
static nvclusterlod_Result makeBoundingSphere(const nvclusterlod_MeshInput& mesh, nvclusterlod_Sphere& sphere)
{
  // TODO: try https://github.com/hbf/miniball
  const nvcluster::vec3f x = fromAPI(ArrayView(mesh.vertexPositions, mesh.vertexCount, mesh.vertexStride)[0]);

  nvcluster::vec3f y = {}, z = {};

  farthestPoint(mesh, x, y);
  farthestPoint(mesh, y, z);

  nvcluster::vec3f position = (y + z) * 0.5f;
  float            radius   = nvcluster::length(z - y) * 0.5f;

  nvcluster::vec3f f = {};
  farthestPoint(mesh, position, f);
  radius = nvcluster::length(f - position);
  if(std::isnan(position[0]) || std::isnan(position[1]) || std::isnan(position[2]) || std::isnan(radius))
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_INCONSISTENT_BOUNDING_SPHERES;
  }

  sphere.center = toAPI(position);
  sphere.radius = radius;

  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

// From a triangle mesh and a partition of its triangles into a number of triangle ranges (DecimatedClusterGroups::groupTriangleRanges), generate a number of clusters within each range
// according to the requested clusterConfig.
static nvclusterlod_Result generateTriangleClusters(nvclusterlod_Context          context,
                                                    const DecimatedClusterGroups& decimatedClusterGroups,
                                                    const nvcluster_Config&       clusterConfig,
                                                    TriangleClusters&             output)
{
  Stopwatch sw(__func__);

  // Compute the bounding boxes and centroids for each triangle
  uint32_t                     triangleCount = decimatedClusterGroups.mesh.triangleCount;
  std::vector<nvcluster_AABB>  triangleAabbs(triangleCount);
  std::vector<vec3>            triangleCentroids(triangleCount);

  NVLOD_PARALLEL_FOR_BEGIN(i, triangleCount, 2048)
  {
    using namespace nvcluster;
    ArrayView   positions(decimatedClusterGroups.mesh.vertexPositions, decimatedClusterGroups.mesh.vertexCount,
                          decimatedClusterGroups.mesh.vertexStride);
    const vec3u triangle = fromAPI(decimatedClusterGroups.mesh.triangleVertices[i]);
    const vec3f a        = fromAPI(positions[triangle[0]]);
    const vec3f b        = fromAPI(positions[triangle[1]]);
    const vec3f c        = fromAPI(positions[triangle[2]]);
    AABB&       aabb     = reinterpret_cast<AABB&>(triangleAabbs[i]);
    aabb.min             = min(a, min(b, c));
    aabb.max             = max(a, max(b, c));

#if 1
    triangleCentroids[i] = aabb.center();
#else
    triangleCentroids[i] = (a + b + c) / 3.0f;
#endif
  }
  NVLOD_PARALLEL_FOR_END;

  // The triangles are now only considered as bounding boxes with a centroid. The segment clusterizer will then
  // generate a number of clusters (each defined by a range in the array of input elements) within each input range (segment)
  // according to the requested clustering configuration.
  nvcluster_Input perTriangleElements{
      .itemBoundingBoxes = triangleAabbs.data(),
      .itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(triangleCentroids.data()),
      .itemCount         = uint32_t(triangleAabbs.size()),
      .itemVertices      = clusterConfig.maxClusterVertices != ~0u ?
                               reinterpret_cast<const uint32_t*>(decimatedClusterGroups.mesh.triangleVertices) :
                               nullptr,
      .vertexCount       = clusterConfig.maxClusterVertices != ~0u ? decimatedClusterGroups.mesh.vertexCount : 0u};

  nvcluster_Result clusteringResult =
      nvcluster::generateSegmentedClusters(context->clusterContext, clusterConfig, perTriangleElements,
                                           nvcluster_Segments{decimatedClusterGroups.groupTriangleRanges.data(),
                                                              uint32_t(decimatedClusterGroups.groupTriangleRanges.size())},
                                           output.clustering);
  if(clusteringResult != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    // FIXME: could translate the clustering error for more details
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTERING_FAILED;
  }

  // For each generated cluster, compute its bounding box so the boxes can be used as input for potential further clustering
  output.clusterAabbs.resize(output.clustering.clusterItemRanges.size());
  NVLOD_PARALLEL_FOR_BEGIN(rangeIndex, output.clusterAabbs.size(), 512)
  {
    const nvcluster_Range& range = output.clustering.clusterItemRanges[rangeIndex];

    nvcluster_AABB clusterAabb = emptyAABB();

    for(uint32_t index = range.offset; index < range.offset + range.count; index++)
    {
      uint32_t triangleIndex = output.clustering.items[index];
      addAABB(clusterAabb, triangleAabbs[triangleIndex]);
    }
    output.clusterAabbs[rangeIndex] = clusterAabb;
  }
  NVLOD_PARALLEL_FOR_END;

  // Store the cluster group index that was used to generate the clusters.
  // FIXME: where is that used?
  output.generatingGroupOffset = decimatedClusterGroups.baseClusterGroupIndex;
  // Store the largest allowed cluster size in the output
  output.maxClusterItems = clusterConfig.maxClusterSize;

  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

// Make clusters of clusters, referred to as "groups", using the cluster adjacency to optimize clustering by keeping
// locked vertices internal to each group (i.e. not on group borders). This is
// important for quality of the recursive decimation.
// This function also sanitizes the cluster adjacency by removing connections involving less than NVLOD_MINIMAL_ADJACENCY_SIZE vertices.
static nvclusterlod_Result groupClusters(nvcluster_Context       context,
                                         const TriangleClusters& triangleClusters,
                                         const nvcluster_Config& clusterGroupConfig,
                                         uint32_t                globalGroupOffset,
                                         ClusterAdjacency&       clusterAdjacency,
                                         ClusterGroups&          result)
{
  Stopwatch sw(__func__);
  using AdjacentCounts = std::unordered_map<uint32_t, AdjacencyVertexCount>;

  // Remove connections between clusters involving less than NVLOD_MINIMAL_ADJACENCY_SIZE vertices, otherwise checkerboard
  // patterns are generated.
  std::vector<uint32_t> adjacencySizes(clusterAdjacency.size(), 0);
  {
    //Stopwatch sw("cleanup");
    NVLOD_PARALLEL_FOR_BEGIN(i, clusterAdjacency.size(), 512)
    {
      AdjacentCounts& adjacency = clusterAdjacency[i];
      for(auto it = adjacency.begin(); it != adjacency.end();)
      {
        if(it->second.vertexCount < NVLOD_MINIMAL_ADJACENCY_SIZE)
        {
          it = adjacency.erase(it);
        }
        else
        {
          ++it;
        }
      }
      adjacencySizes[i] = uint32_t(adjacency.size());
    }
    NVLOD_PARALLEL_FOR_END;
  }

  std::vector<uint32_t> adjacencyOffsets(clusterAdjacency.size(), 0);
  {
    //Stopwatch sw("sum");
    // Get the size of the adjacency list for each cluster (i.e. the number of clusters adjacent to it), and compute the prefix sum of those sizes into adjacencyOffsets.
    // Those offsets will later be used to linearize the adjacency data for the clusters and pass it along for further clustering
    // Note: do NOT use NVLOD_DEFAULT_EXECUTION_POLICY as exclusive_scan seems not to be guaranteed to work in parallel
    std::exclusive_scan(std::execution::seq, adjacencySizes.begin(), adjacencySizes.end(), adjacencyOffsets.begin(), 0,
                        std::plus<uint32_t>());
  }

  // Fill adjacency for clustering input
  // Get the total size of the adjacency list by fetching the offset of the adjacency data of the last cluster and adding the size of its adjacency list
  uint32_t adjacencyItemCount =
      adjacencyOffsets.empty() ? 0u : adjacencyOffsets.back() + uint32_t(clusterAdjacency.back().size());

  // Allocate the buffer storing the linearized per-cluster adjacency data and weights
  std::vector<uint32_t> adjacencyItems(adjacencyItemCount);
  std::vector<float>    adjacencyWeights(adjacencyItemCount);


  // Allocate the buffer storing the ranges within the linearized adjacency buffer corresponding to each cluster
  std::vector<nvcluster_Range> adjacencyRanges(adjacencyOffsets.size());

  std::vector<vec3> clusterCentroids(triangleClusters.clusterAabbs.size());
  // For each cluster, write the adjacency data to the linearized buffer and store the corresponding range for the cluster within that adjacency data
  // and compute cluster centroids as the centroid of their AABBs
  {
    //Stopwatch sw("adj");
    NVLOD_PARALLEL_FOR_BEGIN(clusterIndex, adjacencyOffsets.size(), 512)
    {
      // Initialize the adjacency range with the offset for the cluster, leaving the count to zero and incrementing below
      nvcluster_Range& range  = adjacencyRanges[clusterIndex];
      range                   = {adjacencyOffsets[clusterIndex], 0};

      // Compute the weight of the connection to each adjacent cluster and write the adjacent clusters indices and weights within the range
      for(const auto& [adjacentClusterIndex, adjacencyVertexCounts] : clusterAdjacency[clusterIndex])
      {
        // Compute the weight of the connection, giving more weight to connections with more locked vertices
        float weight = float(1 + adjacencyVertexCounts.vertexCount
                             + adjacencyVertexCounts.lockedCount * NVLOD_LOCKED_VERTEX_WEIGHT_MULTIPLIER);

        // Write the adjacent cluster index and weight to the linearized buffer
        adjacencyItems[range.offset + range.count]   = adjacentClusterIndex;
        adjacencyWeights[range.offset + range.count] = std::max(weight, 1.0f) * NVLOD_VERTEX_WEIGHT_MULTIPLIER;

        // Increment the write position within the range
        range.count++;
      }
      const nvcluster::AABB& clusterAabb = reinterpret_cast<const nvcluster::AABB&>(triangleClusters.clusterAabbs[clusterIndex]);
      clusterCentroids[clusterIndex] = clusterAabb.center();
    }
    NVLOD_PARALLEL_FOR_END;
  }
  // Generate input data for the clusterizer, where the elements to clusterize are the input clusters.
  // We also provide the adjacency data and weights for the clusters to drive the clusterizer, that will
  // attempt to generate graph cuts with minimal weight. Since the weights depend on the number of shared
  // vertices between clusters, the clusterizer will tend to minimize the cost of the graph cuts, hence
  // grouping clusters with more shared vertices.
  nvcluster_Input clusteringInput{
      .itemBoundingBoxes     = triangleClusters.clusterAabbs.data(),
      .itemCentroids         = reinterpret_cast<const nvcluster_Vec3f*>(clusterCentroids.data()),
      .itemCount             = uint32_t(triangleClusters.clusterAabbs.size()),
      .itemConnectionRanges  = adjacencyRanges.data(),
      .connectionTargetItems = adjacencyItems.data(),
      .connectionWeights     = adjacencyWeights.data(),
      .connectionVertexBits  = nullptr,
      .connectionCount       = uint32_t(adjacencyItems.size()),
  };

  result                   = {};
  result.globalGroupOffset = globalGroupOffset;

  nvcluster_Result clusterResult;

  {
    //Stopwatch sw("genclusters");
    clusterResult = nvcluster::generateClusters(context, clusterGroupConfig, clusteringInput, result.groups);
  }
  if(clusterResult != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTERING_FAILED;
  }

  // Compute the total triangle count for each group of clusters of triangles
  result.groupTriangleCounts.resize(result.groups.clusterItemRanges.size(), 0);
  {
    //Stopwatch sw("total");
    for(size_t rangeIndex = 0; rangeIndex < result.groups.clusterItemRanges.size(); rangeIndex++)
    {
      const nvcluster_Range& range = result.groups.clusterItemRanges[rangeIndex];
      for(uint32_t index = range.offset; index < range.offset + range.count; index++)
      {
        uint32_t clusterIndex         = result.groups.items[index];
        uint32_t triangleClusterCount = triangleClusters.clustering.clusterItemRanges[clusterIndex].count;
        result.groupTriangleCounts[rangeIndex] += triangleClusterCount;
        result.totalTriangleCount += triangleClusterCount;
      }
    }
  }
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

static nvclusterlod_Result writeClusters(const DecimatedClusterGroups& decimatedClusterGroups,
                                         ClusterGroups&                clusterGroups,
                                         TriangleClusters&             triangleClusters,
                                         nvclusterlod_MeshOutput&      meshOutput,
                                         OutputWritePositions&         outputWritePositions)
{
  Stopwatch sw(__func__);

  if(outputWritePositions.lodLevelGroup >= meshOutput.lodLevelCount)
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW;
  }

  // Fetch the range of groups for the current LOD level in the output mesh and set its start offset after the last written group count
  nvcluster_Range& lodLevelGroupRange = meshOutput.lodLevelGroupRanges[outputWritePositions.lodLevelGroup];
  outputWritePositions.lodLevelGroup++;
  lodLevelGroupRange.offset = outputWritePositions.groupCluster;

  // Triangle clusters are stored in ranges of the generating group, before
  // decimation. Now that we have re-grouped the triangle clusters into cluster
  // groups we need to track the original generating group per cluster. This
  // saves binary searching to find the generating group index.
  std::vector<uint32_t> clusterGeneratingGroups;
  clusterGeneratingGroups.reserve(triangleClusters.clustering.clusterItemRanges.size());
  for(size_t clusterLocalGroupIndex = 0;
      clusterLocalGroupIndex < triangleClusters.clustering.segmentClusterRanges.size(); clusterLocalGroupIndex++)
  {
    // Fetch the range of clusters corresponding to the current group in the output mesh
    const nvcluster_Range& clusterGroupRange = triangleClusters.clustering.segmentClusterRanges[clusterLocalGroupIndex];
    // For each cluster in the range segment, store the generating group index representing the current segment
    uint32_t generatingGroupIndex = triangleClusters.generatingGroupOffset + uint32_t(clusterLocalGroupIndex);
    clusterGeneratingGroups.insert(clusterGeneratingGroups.end(), clusterGroupRange.count, generatingGroupIndex);
  }

  if(clusterGeneratingGroups.size() != triangleClusters.clustering.clusterItemRanges.size())
  {
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_GENERATING_GROUPS_MISMATCH;
  }

  // Write the clusters to the output
  for(size_t clusterGroupIndex = 0; clusterGroupIndex < clusterGroups.groups.clusterItemRanges.size(); ++clusterGroupIndex)
  {
    // FIXME: check that value
    //if(outputWritePositions.groupCluster >= meshOutput.groupClusterRangeCount)
    if(outputWritePositions.groupCluster >= meshOutput.groupCount)

    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW;
    }
    const nvcluster_Range& range = clusterGroups.groups.clusterItemRanges[clusterGroupIndex];
    std::span<uint32_t> clusterGroup = std::span<uint32_t>(clusterGroups.groups.items.data() + range.offset, range.count);
    meshOutput.groupClusterRanges[outputWritePositions.groupCluster] = {outputWritePositions.clusterTriangleRange, range.count};
    outputWritePositions.groupCluster++;


    // Sort the clusters by their generating group. Clusters are selected based
    // on a comparison between their group and generating group. By storing
    // clusters in contiguous ranges of this intersection, the computation for
    // the whole range can be done at once, or may at least be more cache
    // efficient.
    std::ranges::sort(clusterGroup, [&gg = clusterGeneratingGroups](uint32_t a, uint32_t b) { return gg[a] < gg[b]; });


#if 0
      // Print the generating group membership counts
      std::unordered_map<uint32_t, uint32_t> generatingGroupCounts;
      for (const uint32_t& clusterIndex : clusterGroup)
        generatingGroupCounts[clusterGeneratingGroups[clusterIndex]]++;
      for (auto [gg, count] : generatingGroupCounts)
        printf("Group %zu: generating group %u: clusters: %u\n", clusterGroupIndex + lodLevelGroupRange.offset, gg, count);
#endif

    for(const uint32_t& clusterIndex : clusterGroup)
    {
      nvcluster_Range           clusterTriangleRange = triangleClusters.clustering.clusterItemRanges[clusterIndex];
      std::span<const uint32_t> clusterTriangles =
          std::span<uint32_t>(triangleClusters.clustering.items.data() + clusterTriangleRange.offset,
                              clusterTriangleRange.count);

      const nvclusterlod_Vec3u* trianglesBegin = meshOutput.triangleVertices + outputWritePositions.clusterTriangleVertex;

      uint32_t trianglesBeginIndex = outputWritePositions.clusterTriangleVertex;

      nvcluster_Range clusterRange = {outputWritePositions.clusterTriangleVertex, uint32_t(clusterTriangles.size())};
      if(clusterRange.offset + clusterRange.count > meshOutput.triangleCount)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW;
      }

      // FIXME: reinstate that one
      //assert(outputCounters.clusterTriangleRangeCount < meshOutput.clusterTriangleRangeCount);

      // Gather and write triangles for the cluster. Note these are still global
      // triangle vertex indices. Creating cluster vertices with a vertex cache
      // is intended to be done afterwards.
      for(const uint32_t& triangleIndex : clusterTriangles)
      {
        const nvclusterlod_Vec3u& triangle = decimatedClusterGroups.mesh.triangleVertices[triangleIndex];
        meshOutput.triangleVertices[outputWritePositions.clusterTriangleVertex] = triangle;
        outputWritePositions.clusterTriangleVertex++;
      }

      meshOutput.clusterTriangleRanges[outputWritePositions.clusterTriangleRange] = clusterRange;
      outputWritePositions.clusterTriangleRange++;

      meshOutput.clusterGeneratingGroups[outputWritePositions.clusterParentGroup] = clusterGeneratingGroups[clusterIndex];
      outputWritePositions.clusterParentGroup++;

      // Bounding spheres are an optional output
      if(outputWritePositions.clusterBoundingSphere < meshOutput.clusterCount)
      {
        nvclusterlod_MeshInput mesh{
            .triangleVertices = trianglesBegin,
            .triangleCount    = outputWritePositions.clusterTriangleVertex - trianglesBeginIndex,
            .vertexPositions  = decimatedClusterGroups.mesh.vertexPositions,
            .vertexCount      = decimatedClusterGroups.mesh.vertexCount,
            .vertexStride     = decimatedClusterGroups.mesh.vertexStride,
        };

        nvclusterlod_Result result =
            makeBoundingSphere(mesh, meshOutput.clusterBoundingSpheres[outputWritePositions.clusterBoundingSphere]);
        if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
        {
          return result;
        }
        outputWritePositions.clusterBoundingSphere++;
      }
    }
  }
  lodLevelGroupRange.count = outputWritePositions.groupCluster - lodLevelGroupRange.offset;
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

// TODO: 8 connections per vertex is a lot of memory overhead
struct VertexAdjacency : std::array<uint32_t, 8>
{
  static constexpr const uint32_t Sentinel = 0xffffffffu;
  VertexAdjacency() { std::ranges::fill(*this, Sentinel); }
};

// Returns shared vertex counts between pairs of clusters. The use of the fixed
// sized VertexAdjacency limits cluster vertex valence (not triangle vertex
// valence), but this should be rare for well formed meshes.
static nvclusterlod_Result computeClusterAdjacency(const DecimatedClusterGroups& decimatedClusterGroups,
                                                   const TriangleClusters&       triangleClusters,
                                                   ClusterAdjacency&             result)
{
  Stopwatch sw(__func__);

  // Allocate the cluster connectivity: each cluster will have a map containing the indices of the clusters adjacent to it
  result.resize(triangleClusters.clustering.clusterItemRanges.size());

  // TODO: reduce vertexAdjacency size? overallocated for all vertices in mesh even after decimation

  // For each vertex in the input mesh, we store up to 8 adjacent clusters
  std::vector<VertexAdjacency> vertexClusterAdjacencies(decimatedClusterGroups.mesh.vertexCount);

  // For each triangle cluster, add its cluster index to the adjacency lists of the vertices of the triangles contained in the cluster
  // Each time a vertex is found to be adjacent to another cluster we add the current (resp. other) cluster to the adjacency list of the other (resp. current) cluster,
  // and increment the vertex count for each connection. At the end of this loop we then have, for each cluster, a map of the adjacent clusters indices containing the
  // number of vertices those clusters have in common
  for(uint32_t clusterIndex = 0; clusterIndex < uint32_t(triangleClusters.clustering.clusterItemRanges.size()); ++clusterIndex)
  {
    // Fetch the range of triangles for the current cluster
    const nvcluster_Range& range = triangleClusters.clustering.clusterItemRanges[clusterIndex];
    // Fetch the indices of the triangles contained in the current cluster
    std::span<const uint32_t> clusterTriangles =
        std::span<const uint32_t>(triangleClusters.clustering.items.data() + range.offset, range.count);

    // For each triangle in the cluster, add the current cluster index to the adjacency lists of its vertices
    for(size_t indexInCluster = 0; indexInCluster < clusterTriangles.size(); indexInCluster++)
    {
      // Fetch the current triangle in the cluster
      uint32_t        triangleIndex = clusterTriangles[indexInCluster];
      const vec3u&    tri           = fromAPI(decimatedClusterGroups.mesh.triangleVertices[triangleIndex]);

      // For each vertex of the triangle, add the current cluster index in its adjacency list
      for(uint32_t i = 0; i < 3; ++i)
      {
        // Fetch the cluster adjacency for the vertex
        VertexAdjacency& vertexClusterAdjacency = vertexClusterAdjacencies[tri[i]];
        bool             seenSelf               = false;

        // Check the entries in the adjacency list of the vertex and add the current cluster if not already present
        for(size_t adjacencyIndex = 0; adjacencyIndex < vertexClusterAdjacency.size(); adjacencyIndex++)
        {
          uint32_t& adjacentClusterIndex = vertexClusterAdjacency[adjacencyIndex];

          // If the current cluster has already been added in the vertex adjacency by another triangle there
          // is nothing more to do for that vertex
          if(adjacentClusterIndex == clusterIndex)
          {
            seenSelf = true;
            continue;
          }

          // If we reached the end of the adjacency list and did not find the current cluster index, add
          // the current cluster to the back of the adjacency list
          if(adjacentClusterIndex == VertexAdjacency::Sentinel)
          {
            if(!seenSelf)
            {
              adjacentClusterIndex = clusterIndex;
            }
            if(vertexClusterAdjacency.back() != VertexAdjacency::Sentinel)
            {
              DEBUG_PRINT("Warning: vertexClusterAdjacency[%u] is full\n", tri[i]);
            }
            break;
          }

          // The adjacentIndex is a cluster index, different to the current one,
          // that was previously added and thus is a new connection. Append
          // found connection, once for each direction, and increment the vertex counts for each
          if(adjacentClusterIndex >= clusterIndex)
          {
            return nvclusterlod_Result::NVCLUSTERLOD_ERROR_ADJACENCY_GENERATION_FAILED;
          }
          AdjacencyVertexCount& currentToAdjacent = result[clusterIndex][adjacentClusterIndex];
          AdjacencyVertexCount& adjacentToCurrent = result[adjacentClusterIndex][clusterIndex];
          currentToAdjacent.vertexCount += 1;
          adjacentToCurrent.vertexCount += 1;
          if(decimatedClusterGroups.globalLockedVertices[tri[i]] != 0)
          {
            currentToAdjacent.lockedCount += 1;
            adjacentToCurrent.lockedCount += 1;
          }
        }
      }
    }
  }
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}

// Returns a vector of per-vertex boolean uint8_t values indicating which
// vertices are shared between clusters. Must be uint8_t because that's what
// meshoptimizer takes.
static std::vector<uint8_t> computeLockedVertices(const nvclusterlod_MeshInput& inputMesh,
                                                  const TriangleClusters&       triangleClusters,
                                                  const ClusterGroups&          clusterGrouping)
{
  Stopwatch                sw(__func__);
  constexpr const uint32_t VERTEX_NOT_SEEN = 0xffffffff;
  constexpr const uint32_t VERTEX_ADDED    = 0xfffffffe;
  std::vector<uint8_t>     lockedVertices(inputMesh.vertexCount, 0);
  std::vector<uint32_t>    vertexClusterGroups(inputMesh.vertexCount, VERTEX_NOT_SEEN);
  for(uint32_t clusterGroupIndex = 0; clusterGroupIndex < uint32_t(clusterGrouping.groups.clusterItemRanges.size()); ++clusterGroupIndex)
  {
    const nvcluster_Range&    range = clusterGrouping.groups.clusterItemRanges[clusterGroupIndex];
    std::span<const uint32_t> clusterGroup =
        std::span<const uint32_t>(clusterGrouping.groups.items.data() + range.offset, range.count);
    for(const uint32_t& clusterIndex : clusterGroup)
    {
      const nvcluster_Range&    clusterRange = triangleClusters.clustering.clusterItemRanges[clusterIndex];
      std::span<const uint32_t> cluster =
          std::span<const uint32_t>(triangleClusters.clustering.items.data() + clusterRange.offset, clusterRange.count);
      for(const uint32_t& triangleIndex : cluster)
      {
        const vec3u& tri = fromAPI(inputMesh.triangleVertices[triangleIndex]);
        for(size_t i = 0; i < 3; ++i)
        {
          uint32_t  vertexIndex        = tri[i];
          uint32_t& vertexClusterGroup = vertexClusterGroups[vertexIndex];

          // Initially each vertex is not part of any cluster group. Those are
          // marked with the ID of the first group seen.
          if(vertexClusterGroup == VERTEX_NOT_SEEN)
          {
            vertexClusterGroup = clusterGroupIndex;
          }
          else if(vertexClusterGroup != VERTEX_ADDED && vertexClusterGroup != clusterGroupIndex)
          {
            // Vertex has been seen before and in another cluster group, so it
            // must be shared. VertexAdded is not necessary, but indicates how a
            // unique list of locked vertices might be populated.
            lockedVertices[vertexIndex] = 1;
            vertexClusterGroup          = VERTEX_ADDED;
          }
        }
      }
    }
  }
  return lockedVertices;
}

// Returns groups of triangle after decimating groups of clusters. These
// triangles will be regrouped into new clusters within their current group.
static nvclusterlod_Result decimateClusterGroups(DecimatedClusterGroups& current,
                                                 const TriangleClusters& triangleClusters,
                                                 const ClusterGroups&    clusterGrouping,
                                                 float                   lodLevelDecimationFactor)
{
  Stopwatch                      sw(__func__);
  const nvclusterlod_MeshInput&  inputMesh = current.mesh;
  DecimatedClusterGroups         result;
  // Compute vertices shared between cluster groups. These will be locked during
  // decimation.
  result.globalLockedVertices = computeLockedVertices(inputMesh, triangleClusters, clusterGrouping);
  //FIXME: rethink how DecimatedClusterGroups and InputMesh interact. There must be a way to update the ref to DecimatedClusterGroup
  //bool useOriginalIndices = result.decimatedTriangleStorage.empty();

  result.decimatedTriangleStorage.resize(clusterGrouping.totalTriangleCount);  // space for worst case
  //if (!useOriginalIndices)
  {
    result.mesh.triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(result.decimatedTriangleStorage.data());
  }

  result.groupTriangleRanges.resize(clusterGrouping.groups.clusterItemRanges.size());
  result.groupQuadricErrors.resize(clusterGrouping.groups.clusterItemRanges.size());
  result.baseClusterGroupIndex                 = clusterGrouping.globalGroupOffset;
  std::atomic<uint32_t> decimatedTriangleAlloc = 0;
  nvclusterlod_Result   success                = nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
  NVLOD_PARALLEL_FOR_BEGIN(clusterGroupIndex, clusterGrouping.groups.clusterItemRanges.size(), 1)
  {
    // The cluster group is formed by non-contiguous clusters but the decimator
    // expects contiguous triangle vertex indices. We could reorder triangles by
    // their cluster group, but that would mean reordering the original geometry
    // too. Instead, cluster triangles are flattened into a contiguous vector.

    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      NVLOD_PARALLEL_FOR_BREAK;
    }

    const nvcluster_Range&           clusterGroupRange = clusterGrouping.groups.clusterItemRanges[clusterGroupIndex];
    std::vector<vec3u>               clusterGroupTriangleVertices;
    clusterGroupTriangleVertices.reserve(clusterGroupRange.count * triangleClusters.maxClusterItems);
    for(uint32_t indexInRange = clusterGroupRange.offset;
        indexInRange < clusterGroupRange.offset + clusterGroupRange.count; indexInRange++)
    {
      uint32_t               clusterIndex = clusterGrouping.groups.items[indexInRange];
      const nvcluster_Range& clusterRange = triangleClusters.clustering.clusterItemRanges[clusterIndex];
      for(uint32_t index = clusterRange.offset; index < clusterRange.offset + clusterRange.count; index++)
      {
        uint32_t triangleIndex = triangleClusters.clustering.items[index];

        const vec3u& tri = fromAPI(inputMesh.triangleVertices[triangleIndex]);
        clusterGroupTriangleVertices.push_back(tri);
      }
    }

    // Decimate the cluster group
    std::vector<vec3u>               decimatedTriangleVertices(clusterGroupTriangleVertices.size());
    constexpr float                  targetError   = std::numeric_limits<float>::max();
    float                            absoluteError = 0.0f;
    unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute;  // no meshopt_SimplifyLockBorder as we only care about vertices shared between cluster groups
    size_t desiredTriangleCount = size_t(float(clusterGroupTriangleVertices.size()) * lodLevelDecimationFactor);
    size_t simplifiedTriangleCount =
        meshopt_simplifyWithAttributes(reinterpret_cast<unsigned int*>(decimatedTriangleVertices.data()),
                                       reinterpret_cast<const unsigned int*>(clusterGroupTriangleVertices.data()),
                                       clusterGroupTriangleVertices.size() * 3,
                                       reinterpret_cast<const float*>(inputMesh.vertexPositions), inputMesh.vertexCount,
                                       inputMesh.vertexStride, nullptr, 0, nullptr, 0, result.globalLockedVertices.data(),
                                       desiredTriangleCount * 3, targetError, options, &absoluteError)
        / 3;

    if(desiredTriangleCount < simplifiedTriangleCount)
    {
      DEBUG_PRINT("Warning: decimation failed (%zu < %zu). Retrying, ignoring topology\n", desiredTriangleCount, simplifiedTriangleCount);
      std::vector<vec3u> positionUniqueTriangleVertices(clusterGroupTriangleVertices.size());
      meshopt_generateShadowIndexBuffer(reinterpret_cast<unsigned int*>(positionUniqueTriangleVertices.data()),
                                        reinterpret_cast<const unsigned int*>(clusterGroupTriangleVertices.data()),
                                        clusterGroupTriangleVertices.size() * 3,
                                        reinterpret_cast<const float*>(inputMesh.vertexPositions),
                                        inputMesh.vertexCount, inputMesh.vertexStride, inputMesh.vertexStride);
      simplifiedTriangleCount =
          meshopt_simplifyWithAttributes(reinterpret_cast<unsigned int*>(decimatedTriangleVertices.data()),
                                         reinterpret_cast<const unsigned int*>(positionUniqueTriangleVertices.data()),
                                         positionUniqueTriangleVertices.size() * 3,
                                         reinterpret_cast<const float*>(inputMesh.vertexPositions), inputMesh.vertexCount,
                                         inputMesh.vertexStride, nullptr, 0, nullptr, 0, result.globalLockedVertices.data(),
                                         desiredTriangleCount * 3, targetError, options, &absoluteError)
          / 3;
      if(desiredTriangleCount < simplifiedTriangleCount)
      {
        DEBUG_PRINT("Warning: decimation failed (%zu < %zu). Retrying, ignoring locked\n", desiredTriangleCount, simplifiedTriangleCount);
        simplifiedTriangleCount =
            meshopt_simplifySloppy(reinterpret_cast<unsigned int*>(decimatedTriangleVertices.data()),
                                   reinterpret_cast<const unsigned int*>(positionUniqueTriangleVertices.data()),
                                   positionUniqueTriangleVertices.size() * 3,
                                   reinterpret_cast<const float*>(inputMesh.vertexPositions), inputMesh.vertexCount,
                                   inputMesh.vertexStride, desiredTriangleCount * 3, targetError, &absoluteError)
            / 3;
      }
    }

    // Handle a case when meshopt_simplifySloppy() sometimes returns no
    // triangles for very sparse geometry.
    if(simplifiedTriangleCount == 0)
    {
      DEBUG_PRINT("Warning: decimation produced no triangles. Adding the first back\n");
      simplifiedTriangleCount++;
      decimatedTriangleVertices[0] = clusterGroupTriangleVertices[0];
    }

    // HACK: truncate triangles if decimation target was not met
    if(desiredTriangleCount < simplifiedTriangleCount)
    {
      DEBUG_PRINT("Warning: decimation failed (%zu < %zu). Discarding %zu triangles\n", desiredTriangleCount,
                  simplifiedTriangleCount, simplifiedTriangleCount - desiredTriangleCount);

#if WRITE_DECIMATION_FAILURE_OBJS
      auto writeObjGeometry = [](std::ostream& os, auto& triangles, auto& positions) {
        os << "g mesh\n";
        for(auto& p : positions)
          os << "v " << p.x << " " << p.y << " " << p.z << "\n";
        for(auto& t : triangles)
          os << "f " << t.x + 1 << " " << t.y + 1 << " " << t.z + 1 << "\n";
      };
      static std::atomic<int> i = 0;
      std::ofstream           f("failure" + std::to_string(i++) + ".obj");
      writeObjGeometry(f, clusterGroupTriangleVertices,
                       ArrayView(inputMesh.vertexPositions, inputMesh.vertexCount, inputMesh.vertexStride));
#endif
    }
    decimatedTriangleVertices.resize(std::min(desiredTriangleCount, simplifiedTriangleCount));

    // Allocate output for this thread
    uint32_t groupDecimatedTrianglesOffset = decimatedTriangleAlloc.fetch_add(uint32_t(decimatedTriangleVertices.size()));

    // Copy decimated triangle indices to result.decimatedTriangleStorage. This
    // temporary buffer is needed and we can't write directly to the library
    // user's buffer because triangles must be ordered by cluster, which is
    // computed next.
    if(groupDecimatedTrianglesOffset + decimatedTriangleVertices.size() > result.decimatedTriangleStorage.size())
    {
      success = nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_MESH_OVERFLOW;
      NVLOD_PARALLEL_FOR_BREAK;
    }
    std::ranges::copy(decimatedTriangleVertices, result.decimatedTriangleStorage.begin() + groupDecimatedTrianglesOffset);

    result.groupTriangleRanges[clusterGroupIndex] = {uint32_t(groupDecimatedTrianglesOffset),
                                                     uint32_t(decimatedTriangleVertices.size())};
    result.groupQuadricErrors[clusterGroupIndex]  = absoluteError;
  }
  NVLOD_PARALLEL_FOR_END;

  if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
  {
    return success;
  }

  result.decimatedTriangleStorage.resize(decimatedTriangleAlloc);
  result.mesh.triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(result.decimatedTriangleStorage.data());
  result.mesh.triangleCount    = uint32_t(result.decimatedTriangleStorage.size());
  result.mesh.vertexCount      = inputMesh.vertexCount;
  result.mesh.vertexStride = inputMesh.vertexStride;
  result.mesh.vertexPositions  = inputMesh.vertexPositions;

  std::swap(current, result);
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}
}  // namespace nvclusterlod

// Returns the ceiling of an integer division.
static uint32_t divCeil(const uint32_t& a, const uint32_t& b)
{
  return (a + b - 1) / b;
}

nvclusterlod_Result nvclusterlodGetMeshRequirements(nvclusterlod_Context /*context*/,
                                                    const nvclusterlod_MeshInput* input,
                                                    nvclusterlod_MeshCounts*      outputRequiredCounts)
{
  uint32_t triangleCount = input->triangleCount;
  uint32_t minClusterSize = input->clusterConfig.minClusterSize;
  if(input->clusterConfig.maxClusterVertices != ~0u)
  {
    // Using a vertex limit reduces the minimum cluster size to 1, resulting in
    // a very large worst case of one cluster per triangle even though we rarely
    // get close.
    minClusterSize = 1;
  }

  assert(triangleCount != 0);
  uint32_t lod0ClusterCount       = divCeil(triangleCount, minClusterSize) + 1;
  uint32_t idealLevelCount        = uint32_t(ceilf(-logf(float(lod0ClusterCount)) / logf(input->decimationFactor)));
  uint32_t idealClusterCount = lod0ClusterCount * idealLevelCount;
  uint32_t idealClusterGroupCount = divCeil(idealClusterCount, input->groupConfig.minClusterSize);

  // TODO: actually validate against overflow
  *outputRequiredCounts = nvclusterlod_MeshCounts{
      .triangleCount = idealClusterCount * minClusterSize,
      .clusterCount  = idealClusterCount,
      .groupCount    = idealClusterGroupCount * 4,  // DANGER: group min-cluster-count is less than the max
      .lodLevelCount = idealLevelCount * 2 + 1,     // "* 2 + 1" - why is this needed
  };
  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}


nvclusterlod_Result nvclusterlodBuildMesh(nvclusterlod_Context context, const nvclusterlod_MeshInput* inputPtr, nvclusterlod_MeshOutput* output)
{
  Stopwatch sw(__func__);

  const nvclusterlod_MeshInput& input = *inputPtr;

  if(input.clusterConfig.maxClusterVertices != ~0u && input.clusterConfig.itemVertexCount != 3)
  {
    // User wants to set a vertex limit. Only triangles are supported, which
    // have 3 vertices.
    return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_ITEM_VERTEX_COUNT_NOT_THREE;
  }

  nvclusterlod::OutputWritePositions outputCounters{};

  // Populate initial mesh input in a common structure. Subsequent passes
  // contain results from the previous level of detail.
  nvclusterlod::DecimatedClusterGroups decimatedClusterGroups{
      .groupTriangleRanges = {{0, input.triangleCount}},  // The first pass uses the entire input mesh, hence we only have one large group of triangles
      .mesh = input,
      .decimatedTriangleStorage = {},  // initial source is input.mesh so this is empty. Further passes will write the index buffer for the decimated triangles of the LODs
      .groupQuadricErrors = {0.0f},    // In the first pass no error has yet been accumulated
      .baseClusterGroupIndex = NVCLUSTERLOD_ORIGINAL_MESH_GROUP,  // The first group represents the original mesh, hence we mark it as the original group
      .globalLockedVertices = std::vector<uint8_t>(input.vertexCount, 0),  // No vertices are locked in the original mesh
  };

  // Initial clustering
  DEBUG_PRINT("Initial clustering (%u triangles)\n", input.triangleCount);

#if PRINT_OPS
  uint32_t lodLevel = 0;
#endif
  // Loop creating LOD levels until there is a single root cluster.
  size_t lastTriangleCount   = std::numeric_limits<size_t>::max();
  int    triangleCountCanary = 10;
  while(true)
  {
    // Cluster the initial or decimated geometry. When clustering decimated
    // geometry, clusters are only formed within groups of triangles from the
    // last iteration.

    // In the first iteration (LOD 0) the mesh is represented by a single range of triangles covering the entire mesh. The
    // function generateTriangleClusters will create a set of clusters from this mesh. Each cluster is represented by
    // a range of triangles within the mesh.
    // Later iterations will take the clusters from the previous iteration as input. The function generateTriangleClusters
    // will then create a set of clusters within each of the input clusters.
    nvclusterlod::TriangleClusters triangleClusters{};
    nvclusterlod_Result success = generateTriangleClusters(context, decimatedClusterGroups, input.clusterConfig, triangleClusters);
    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      return success;
    }

    // Compute the adjacency between clusters: for each cluster clusterAdjacency will contain a map of
    // its adjacent clusters along with the number of vertices shared with each of those clusters. The adjacency information is symmetric.
    // This is important as it feeds into the weights for making groups of clusters.
    nvclusterlod::ClusterAdjacency clusterAdjacency{};
    success = computeClusterAdjacency(decimatedClusterGroups, triangleClusters, clusterAdjacency);
    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      return success;
    }


    // Make clusters of clusters, called "cluster groups" or just "groups".
    uint32_t                    globalGroupOffset = outputCounters.groupCluster;
    nvclusterlod::ClusterGroups clusterGroups{};
    success = groupClusters(context->clusterContext, triangleClusters, input.groupConfig, globalGroupOffset,
                            clusterAdjacency, clusterGroups);
    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      return success;
    }

    // Write the generated clusters and cluster groups representing the mesh at the current LOD
    success = nvclusterlod::writeClusters(decimatedClusterGroups, clusterGroups, triangleClusters, *output, outputCounters);
    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      return success;
    }


    // Exit when there is just one cluster, meaning the decimation reached the level where the entire mesh geometry fits within a single cluster
    uint32_t clusterCount = uint32_t(triangleClusters.clustering.clusterItemRanges.size());
    if(clusterCount <= 1)
    {
      if(clusterCount != 1)
      {
        return nvclusterlod_Result::NVCLUSTERLOD_ERROR_EMPTY_ROOT_CLUSTER;
      }
      break;
    }

    // Decimate within cluster groups to create the next LOD level
    DEBUG_PRINT("Decimating lod %d (%d clusters)\n", lodLevel++, clusterCount);

    float maxDecimationFactor = float(clusterCount - 1) / float(clusterCount);
    float decimationFactor    = std::min(maxDecimationFactor, input.decimationFactor);
    success = decimateClusterGroups(decimatedClusterGroups, triangleClusters, clusterGroups, decimationFactor);

    if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    {
      return success;
    }


    // Make sure the number of triangles is always going down. This may fail for
    // high decimation factors.
    size_t triangleCount = decimatedClusterGroups.decimatedTriangleStorage.size();
    if(triangleCount == lastTriangleCount && --triangleCountCanary <= 0)
    {
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_CLUSTER_COUNT_NOT_DECREASING;
    }
    lastTriangleCount = triangleCount;

    // Per-group quadric errors are written separately as the final LOD level
    // of groups will not decimate and need zeroes written instead.
    for(size_t i = 0; i < decimatedClusterGroups.groupQuadricErrors.size(); i++)
    {
      output->groupQuadricErrors[outputCounters.groupQuadricError] = decimatedClusterGroups.groupQuadricErrors[i];
      outputCounters.groupQuadricError++;
    }
  }

  // Write zeroes for the final LOD level of groups (of which there is only
  // one), which do not decimate
  // TODO: shouldn't this be infinite error so it's always drawn?
  output->groupQuadricErrors[outputCounters.groupQuadricError] = 0.f;
  outputCounters.groupQuadricError++;

  output->clusterCount  = outputCounters.clusterTriangleRange;
  output->groupCount    = outputCounters.groupCluster;
  output->lodLevelCount = outputCounters.lodLevelGroup;
  output->triangleCount = outputCounters.clusterTriangleVertex;


  return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
}
