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

#include <cstdint>

#include "nvcluster/nvcluster.h"
#include "nvclusterlod_common.h"


namespace nvclusterlod {

// Each level of detail is represented by a group of clusters. This value identifies the original group, corresponding to the entire input mesh
const uint32_t ORIGINAL_MESH_GROUP = ~0u;

// High density mesh and clustering parameters used to generate the LODs for the mesh
struct MeshInput
{
  // Pointer to triangle definitions, 3 indices per triangle
  const uint32_t* indices = nullptr;
  // Number of indices in the mesh
  uint32_t indexCount = 0u;

  // Vertex data for the mesh, 3 floats per entry
  const float* vertices = nullptr;
  // Offset in vertices where the vertex data for the mesh starts, in float
  uint32_t vertexOffset = 0u;
  // Number of vertices in the mesh
  uint32_t vertexCount = 0u;
  // Stride in bytes between the beginning of two successive vertices (e.g. 12 bytes for densely packed positions)
  uint32_t vertexStride = 0u;

  // Configuration for the generation of triangle clusters
  nvcluster::Config clusterConfig = {
      .minClusterSize    = 96,
      .maxClusterSize    = 128,
      .costUnderfill     = 0.9f,
      .costOverlap       = 0.5f,
      .preSplitThreshold = 1u << 17,
  };

  // Configuration for the generation of cluster groups
  // Each LOD is comprised of a number of cluster groups
  nvcluster::Config groupConfig = {
      .minClusterSize    = 24,
      .maxClusterSize    = 32,
      .costUnderfill     = 0.5f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  // Decimation factor applied between successive LODs
  float decimationFactor = 0.f;
};

// Memory requirements for the storage of the entirety of the mesh, including its LODs
struct MeshRequirements
{
  // Maximum total number of triangles across LODs
  uint32_t maxTriangleCount = 0u;
  // Maximum total number of clusters across LODs
  uint32_t maxClusterCount = 0u;
  // Maximum total number of cluster groups across LODs
  uint32_t maxGroupCount = 0u;
  // Maximum number of LODs in the mesh
  uint32_t maxLodLevelCount = 0u;
};

struct MeshOutput
{
  // Triangle clusters. Each Range represents one cluster covering range.count triangles in clusterTriangles, starting at range.offset
  nvcluster::Range* clusterTriangleRanges = nullptr;

  // Triangle data for the clusters, referenced by clusterTriangleRanges
  uint32_t* clusterTriangles = nullptr;

  // Decimation takes the mesh at a given LOD represented by a number of cluster groups,
  // and generates a (smaller) number of cluster groups for the next coarser LOD. For each
  // generated cluster clusterGeneratingGroups stores the index of the group it was generated from.
  // For the clusters at the finest LOD (LOD 0) that index is ORIGINAL_MESH_GROUP
  uint32_t* clusterGeneratingGroups = nullptr;

  // Bounding spheres of the clusters, may be nullptr
  Sphere* clusterBoundingSpheres = nullptr;

  // Error metric after decimating geometry in each group. Counter-intuitively,
  // not the error of the geometry in the group - that value does not exist
  // per-group but would be the group quadric error of the cluster's generating
  // group. This saves duplicating errors per cluster. The final LOD is just one
  // group, is not decimated, and has an error of zero.
  // TODO: shouldn't this be infinite error so it's always drawn?
  float* groupQuadricErrors = nullptr;

  // Ranges of clusters contained in each group so that the clusters of a group are stored at range.offset in clusterTriangleRanges
  // and the group covers range.count clusters.
  nvcluster::Range* groupClusterRanges = nullptr;

  // Ranges of groups comprised in each LOD level, so that the groups for LOD n are stored at lodLevelGroupRanges[n].offset and the LOD
  // uses lodLevelGroupRanges[n].count groups. The finest LOD is at index 0, followed by the coarser LODs from finer to coarser
  nvcluster::Range* lodLevelGroupRanges = nullptr;

  // Number of triangles in the mesh across LODs
  uint32_t triangleCount = 0u;
  // Number of clusters in the mesh across LODs
  uint32_t clusterCount = 0u;
  // Number of cluster groups in the mesh across LODs
  uint32_t groupCount = 0u;
  // Number of LODs in the mesh
  uint32_t lodLevelCount = 0u;
};

// Structure to request the memory requirements to build the LODs for the input mesh
struct MeshGetRequirementsInfo
{
  // Definition of the input geometry and clustering configuration
  const MeshInput* input = nullptr;
};

// Structure to build the LODs for the input mesh
struct MeshCreateInfo
{
  // Definition of the input geometry and clustering configuration
  const MeshInput* input = nullptr;
};

}  // namespace nvclusterlod

extern "C" {

// Usage:
// 1. call nvlodMeshGetRequirements(...) to get conservative sizes
// 2. allocate Output data
// 3. call nvlodMeshCreate(...)
// 4. resize down to what was written
// Alternatively use nvlod::LodMesh, which encapsulates the above
// Note that returned vertices are global indices. The utility
// nvlod::LocalizedLodMesh can be used to create vertex indices local
// to each cluster.

// Request the memory requirements to build the LODs for the input mesh
nvclusterlod::MeshRequirements nvclusterlodMeshGetRequirements(nvclusterlod::Context                        context,
                                                               const nvclusterlod::MeshGetRequirementsInfo* info);

// Build the LODs for the input mesh
nvclusterlod::Result nvclusterlodMeshCreate(nvclusterlod::Context               context,
                                            const nvclusterlod::MeshCreateInfo* info,
                                            nvclusterlod::MeshOutput*           output);
}