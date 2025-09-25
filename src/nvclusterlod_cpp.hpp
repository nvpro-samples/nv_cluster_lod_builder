/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <array_view.hpp>
#include <cassert>
#include <nvcluster/util/objects.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <span>

namespace nvclusterlod {

// Output span plus a write index
template <typename T>
class OutputSpan
{
public:
  OutputSpan(std::span<T> output_)
      : output(output_)
  {
  }
  uint32_t     capacity() const { return static_cast<uint32_t>(output.size()); }
  T&           allocate() { return output[writeIndex++]; }
  std::span<T> allocate(uint32_t count)
  {
    auto result = std::span(output).subspan(writeIndex, count);
    writeIndex += count;
    return result;
  }
  uint32_t allocatedCount() const { return writeIndex; }
  bool     full() const { return writeIndex == output.size(); }
  void     append(const T& value) { output[writeIndex++] = value; }
  void     append(const std::span<const T>& values)
  {
    std::copy(values.begin(), values.end(), output.begin() + writeIndex);
    writeIndex += static_cast<uint32_t>(values.size());
  }
  std::span<const T> allocated() const { return std::span(output).subspan(0, writeIndex); }

private:
  std::span<T> output;
  uint32_t     writeIndex = 0u;
};

// Sphere with a vec3f center that's binary compatible with nvclusterlod_Sphere
struct Sphere
{
  nvcluster::vec3f center;
  float            radius;
  operator nvclusterlod_Sphere() const { return {center, radius}; }
};
static_assert(sizeof(Sphere) == sizeof(nvclusterlod_Sphere));

// Returns whether `inner` is inside or equal to `outer`.
inline bool isInside(const Sphere& inner, const Sphere& outer)
{
  const float radiusDifference = outer.radius - inner.radius;
  return (radiusDifference >= 0.0f)  // if this is negative then `inner` cannot be inside `outer`
         && length_squared(inner.center - outer.center) <= radiusDifference * radiusDifference;
}

struct Mesh
{
  static Mesh fromCAPI(const nvclusterlod_MeshInput& input)
  {
    return {.triangleVertices = {reinterpret_cast<const nvcluster::vec3u*>(input.triangleVertices), input.triangleCount},
            .vertexPositions = {reinterpret_cast<const nvcluster::vec3f*>(input.vertexPositions), input.vertexCount,
                                input.vertexStride}};
  }
  std::span<const nvcluster::vec3u> triangleVertices;
  ArrayView<const nvcluster::vec3f> vertexPositions;
};

// Safer nvclusterlod_MeshInput wrapper with bounds checkable spans around C API
struct MeshInput
{
  MeshInput(const nvclusterlod_MeshInput& input)
      : mesh(Mesh::fromCAPI(input))
      , capi(input)
  {
  }
  Mesh                          mesh;
  const nvclusterlod_MeshInput& capi;
};

// Safer nvclusterlod_MeshOutput wrapper with bounds checkable spans around C API
struct MeshOutput
{
  MeshOutput(const nvclusterlod_MeshOutput& output)
      : clusterTriangleRanges({reinterpret_cast<nvcluster::Range*>(output.clusterTriangleRanges), output.clusterCount})
      , triangleVertices({reinterpret_cast<nvcluster::vec3u*>(output.triangleVertices), output.triangleCount})
      , clusterGeneratingGroups({output.clusterGeneratingGroups, output.clusterCount})
      , clusterBoundingSpheres({reinterpret_cast<Sphere*>(output.clusterBoundingSpheres),
                                output.clusterBoundingSpheres ? output.clusterCount : 0u})
      , groupQuadricErrors({output.groupQuadricErrors, output.groupCount})
      , groupClusterRanges({reinterpret_cast<nvcluster::Range*>(output.groupClusterRanges), output.groupCount})
      , lodLevelGroupRanges({reinterpret_cast<nvcluster::Range*>(output.lodLevelGroupRanges), output.lodLevelCount})
  {
  }

  nvclusterlod_Result writeCounts(nvclusterlod_MeshOutput& output)
  {
    // Triangle count
    output.triangleCount = triangleVertices.allocatedCount();

    // Cluster count
    output.clusterCount = clusterTriangleRanges.allocatedCount();
    if(output.clusterCount != clusterGeneratingGroups.allocatedCount())
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_INCONSISTENT_COUNTS;
    if(clusterBoundingSpheres.capacity() && output.clusterCount != clusterBoundingSpheres.allocatedCount())
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_INCONSISTENT_COUNTS;

    // Group count
    output.groupCount = groupQuadricErrors.allocatedCount();
    if(output.groupCount != groupClusterRanges.allocatedCount())
      return nvclusterlod_Result::NVCLUSTERLOD_ERROR_OUTPUT_INCONSISTENT_COUNTS;

    // LOD level count
    output.lodLevelCount = lodLevelGroupRanges.allocatedCount();
    return nvclusterlod_Result::NVCLUSTERLOD_SUCCESS;
  }

  OutputSpan<nvcluster::Range> clusterTriangleRanges;
  OutputSpan<nvcluster::vec3u> triangleVertices;
  OutputSpan<uint32_t>         clusterGeneratingGroups;
  OutputSpan<Sphere>           clusterBoundingSpheres;
  OutputSpan<float>            groupQuadricErrors;
  OutputSpan<nvcluster::Range> groupClusterRanges;
  OutputSpan<nvcluster::Range> lodLevelGroupRanges;
};

// Safer nvclusterlod_HierarchyInput wrapper with bounds-checkable spans
struct HierarchyInput
{
  static HierarchyInput fromCAPI(const nvclusterlod_HierarchyInput& input)
  {
    return {
        .clusterGeneratingGroups = {input.clusterGeneratingGroups, input.clusterCount},
        .clusterBoundingSpheres  = {reinterpret_cast<const Sphere*>(input.clusterBoundingSpheres), input.clusterCount},
        .groupQuadricErrors      = {input.groupQuadricErrors, input.groupCount},
        .groupClusterRanges = {reinterpret_cast<const nvcluster::Range*>(input.groupClusterRanges), input.groupCount},
        .lodLevelGroupRanges = {reinterpret_cast<const nvcluster::Range*>(input.lodLevelGroupRanges), input.lodLevelCount},
    };
  }

  std::span<const uint32_t>         clusterGeneratingGroups;
  std::span<const Sphere>           clusterBoundingSpheres;  // may be empty if not provided
  std::span<const float>            groupQuadricErrors;
  std::span<const nvcluster::Range> groupClusterRanges;
  std::span<const nvcluster::Range> lodLevelGroupRanges;
};

// Safer nvclusterlod_HierarchyOutput wrapper with bounds-checkable spans
struct HierarchyOutput
{
  // groupCount is required to size the per-group outputs, as the C API does not
  // carry these capacities in the output struct; it is implied from the input.
  HierarchyOutput(const nvclusterlod_HierarchyOutput& output, uint32_t groupCount)
      : nodes({reinterpret_cast<nvclusterlod_HierarchyNode*>(output.nodes), output.nodeCount})
      , groupCumulativeBoundingSpheres{reinterpret_cast<Sphere*>(output.groupCumulativeBoundingSpheres), groupCount}
      , groupCumulativeQuadricError{output.groupCumulativeQuadricError, groupCount}
  {
  }

  void writeCounts(nvclusterlod_HierarchyOutput& output) { output.nodeCount = nodes.allocatedCount(); }

  OutputSpan<nvclusterlod_HierarchyNode> nodes;
  std::span<Sphere>                      groupCumulativeBoundingSpheres;
  std::span<float>                       groupCumulativeQuadricError;
};

}  // namespace nvclusterlod
