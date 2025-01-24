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

#include <cmath>
#include <cstdint>

#include "nvclusterlod_common.h"
#include <nvcluster/nvcluster.h>

namespace nvclusterlod {

// Input structure for the generation of hierarchical LODs on an input defined by its LODs,
// where each LOD is defined by a range of cluster groups, each of those groups being defined
// by a range of clusters. Each cluster is characterized by its bounding sphere.
struct HierarchyInput
{
  // Decimation takes the mesh at a given LOD represented by a number of cluster groups,
  // and generates a (smaller) number of cluster groups for the next coarser LOD. For each
  // generated cluster clusterGeneratingGroups stores the index of the group it was generated from.
  // For the clusters at the finest LOD (LOD 0) that index is ORIGINAL_MESH_GROUP
  const uint32_t* clusterGeneratingGroups = nullptr;

  // Error metric after decimating geometry in each group
  const float* groupQuadricErrors = nullptr;

  // Ranges of clusters contained in each group so that the clusters of a group are stored at range.offset
  // and the group covers range.count clusters.
  const nvcluster::Range* groupClusterRanges = nullptr;

  // Number of cluster groups
  uint32_t groupCount = 0u;


  // Bounding sphere for each cluster
  const Sphere* clusterBoundingSpheres = nullptr;
  // Number of clusters
  uint32_t clusterCount = 0u;


  // Ranges of groups comprised in each LOD level, so that the groups for LOD n are stored at lodLevelGroupRanges[n].offset and the LOD
  // uses lodLevelGroupRanges[n].count groups. The finest LOD is at index 0, followed by the coarser LODs from finer to coarser
  const nvcluster::Range* lodLevelGroupRanges = nullptr;

  // Number of LODs in the mesh
  uint32_t lodLevelCount = 0u;

  // Enforce a minimum LOD rate of change. This is the maximum sine of the error
  // angle threshold that will be used to compute LOD for a given camera
  // position. See Output::maxQuadricErrorOverDistance and
  // pixelErrorToQuadricErrorOverDistance(). Increase this if LOD levels
  // overlap.
  float minQuadricErrorOverDistance = 0.001f;

  // Bounding spheres include the bounding spheres of generating groups. This
  // guarantees no overlaps regardless of the error over distance threshold.
  bool conservativeBoundingSpheres = true;
};

// Interior node in the LOD DAG
struct InteriorNode
{
  // Maximum number of children for the node
  static const uint32_t NODE_RANGE_MAX_SIZE = 32;
  // Either InteriorNode or LeafNode can be stored in Node, isLeafNode will be 0 for InteriorNode
  uint32_t isLeafNode : 1;
  // Offset in FIXME where the children of the node can be found
  uint32_t childOffset : 26;
  // Number of children for the node, minus one as the children list of an interior node contains at least a leaf node
  // representing its geometry at its corresponding LOD
  uint32_t childCountMinusOne : 5;
};
static_assert(sizeof(InteriorNode) == sizeof(uint32_t));

// Leaf node in the LOD DAG
struct LeafNode
{
  static const uint32_t CLUSTER_RANGE_MAX_SIZE = 256;
  // Either InteriorNode or LeafNode can be stored in Node, isLeafNode will be 1 for LeafNode
  uint32_t isLeafNode : 1;  // clusterGroupNode?
  // Index of the cluster group for the node
  uint32_t group : 23;
  // Number of clusters in the group, minus one as a group always contains at least one cluster
  uint32_t clusterCountMinusOne : 8;
};
static_assert(sizeof(LeafNode) == sizeof(uint32_t));

// LOD DAG node
struct Node
{
  // Node definition, either interior or leaf node
  union
  {
    InteriorNode children;
    LeafNode     clusters;
  };

  // Bounding sphere for the node
  Sphere boundingSphere;

  // Maximum error due to the mesh decimation at the LOD of the node
  float maxClusterQuadricError = 0.f;
};

// Output structure for the LOD hierarchy
struct HierarchyOutput
{
  // LOD DAG
  Node* nodes = nullptr;
  // Bounding sphere for each cluster group, encompassing all the clusters within the group
  Sphere* groupCumulativeBoundingSpheres = nullptr;
  // Quadric errors obtained by accumulating the quadric errors of the clusters within the group
  float* groupCumulativeQuadricError = nullptr;
  // Number of nodes in the DAG
  uint32_t nodeCount = 0u;
};

// Structure to request the memory requirements to build a LOD DAG on the input data
struct HierarchyGetRequirementsInfo
{
  const HierarchyInput* input = nullptr;
};

// Memory requirements to build a LOD DAG
struct HierarchyRequirements
{
  uint32_t maxNodeCount = 0u;
};

// Structure to request to build a LOD DAG on the input data
struct HierarchyCreateInfo
{
  const HierarchyInput* input = nullptr;
};

}  // namespace nvclusterlod

extern "C" {
// Get the memory requirements to generate the LOD DAG for the input
nvclusterlod::HierarchyRequirements nvclusterlodHierarchyGetRequirements(nvclusterlod::Context context,
                                                                         const nvclusterlod::HierarchyGetRequirementsInfo* info);

// Generate the LOD DAG for the input
nvclusterlod::Result nvclusterlodCreateHierarchy(nvclusterlod::Context                    context,
                                                 const nvclusterlod::HierarchyCreateInfo* info,
                                                 nvclusterlod::HierarchyOutput*           output);
}