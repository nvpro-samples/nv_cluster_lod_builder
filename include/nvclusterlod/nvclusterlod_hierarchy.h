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
#ifndef NVCLUSTERLOD_HIERARCHY_H
#define NVCLUSTERLOD_HIERARCHY_H

#include <math.h>
#include <nvcluster/nvcluster.h>
#include <nvclusterlod/nvclusterlod_common.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The hierarchy input is mostly nvclusterlod_MeshOutput with const pointers,
// with the exception of triangle data that is not needed.
typedef struct nvclusterlod_HierarchyInput
{
  // The group of clusters that was decimated to produce the geometry in each
  // cluster, or NVCLUSTERLOD_ORIGINAL_MESH_GROUP if the cluster is original
  // mesh geometry. This relationship forms a DAG. Levels of detail are
  // generated by iteratively decimating groups of clusters and re-clustering
  // the result. The clusters in a group will have mixed generating groups. See
  // the readme for a visuzliation.
  const uint32_t* clusterGeneratingGroups NVCLUSTERLOD_DEFAULT(nullptr);

  // Bounding spheres of the clusters, may be nullptr
  // TODO: verify 'may be nullptr' - likely doesn't work anymore
  const nvclusterlod_Sphere* clusterBoundingSpheres NVCLUSTERLOD_DEFAULT(nullptr);

  // Error metric after decimating geometry in each group. Counter-intuitively,
  // not the error of the geometry in the group - that value does not exist
  // per-group. For the current level, use
  // groupQuadricErrors[clusterGeneratingGroups[cluster]]. This saves
  // duplicating data per cluster. The final LOD (just one
  // group) is not decimated and has an error of zero.
  // TODO: shouldn't this be infinite error so it's always drawn?
  const float* groupQuadricErrors NVCLUSTERLOD_DEFAULT(nullptr);

  // Ranges of clusters for each group of clusters. I.e. cluster values for a
  // group are stored at cluster*[range.offset + i] for i in {0 .. range.count -
  // 1}.
  const nvcluster_Range* groupClusterRanges NVCLUSTERLOD_DEFAULT(nullptr);

  // Ranges of groups for each LOD level. I.e. group values for a LOD are stored
  // at group*[range.offset + i] for i in {0 .. range.count - 1}. The finest LOD
  // is at index 0 (comprised of clusters of the original mesh), followed by the
  // coarser LODs from finer to coarser.
  const nvcluster_Range* lodLevelGroupRanges NVCLUSTERLOD_DEFAULT(nullptr);

  // Number of clusters for all LODs
  uint32_t clusterCount NVCLUSTERLOD_DEFAULT(0u);

  // Number of cluster groups for all LODs
  uint32_t groupCount NVCLUSTERLOD_DEFAULT(0u);

  // Number of LOD levels
  uint32_t lodLevelCount NVCLUSTERLOD_DEFAULT(0u);
} nvclusterlod_HierarchyInput;

// Limit imposed by the bits in nvclusterlod_InternalNodeChildren::childCountMinusOne
#define NVCLUSTERLOD_NODE_MAX_CHILDREN (32u)

// Packed children subrange used in interior nodes in the spatial hierarchy of
// bounding spheres
typedef struct nvclusterlod_InternalNodeChildren
{
  // A nvclusterlod_HierarchyNode is one of the following, both overlaying this
  // single bit type enum to conserve space:
  // - 0: nvclusterlod_InternalNodeChildren
  // - 1: nvclusterlod_LeafNodeClusterGroup
  uint32_t isClusterGroup : 1;

  // Offset of the first child in the nodes array
  uint32_t childOffset : 26;

  // Number of children, minus one to reclaim the value 0, which would be
  // invalid
  uint32_t childCountMinusOne : 5;
} nvclusterlod_InternalNodeChildren;

NVCLUSTERLOD_STATIC_ASSERT(sizeof(nvclusterlod_InternalNodeChildren) == sizeof(uint32_t))

// Limit imposed by the bits in nvclusterlod_LeafNodeClusterGroup::clusterCountMinusOne
#define NVCLUSTERLOD_GROUP_MAX_CLUSTERS (256u)

// Cluster group referenced by leaf nodes in the spatial hierarchy of bounding
// spheres
typedef struct nvclusterlod_LeafNodeClusterGroup
{
  // A nvclusterlod_HierarchyNode is one of the following, both overlaying this
  // single bit type enum to conserve space:
  // - 0: nvclusterlod_InternalNodeChildren
  // - 1: nvclusterlod_LeafNodeClusterGroup
  uint32_t isClusterGroup : 1;

  // Index of the cluster group for the node
  uint32_t group : 23;

  // Number of clusters in the group, minus one to reclaim 0 as a group always
  // contains at least one cluster
  uint32_t clusterCountMinusOne : 8;
} nvclusterlod_LeafNodeClusterGroup;

NVCLUSTERLOD_STATIC_ASSERT(sizeof(nvclusterlod_LeafNodeClusterGroup) == sizeof(uint32_t))

// Spatial hierarchy node
typedef struct nvclusterlod_HierarchyNode
{
  // May either point to more children or a cluster group. The single bit type
  // enum isClusterGroup is aliased in both.
  union
  {
    nvclusterlod_InternalNodeChildren children;
    nvclusterlod_LeafNodeClusterGroup clusterGroup;
  };

  // Bounding sphere for the node. Note the bounding sphere conservatively and
  // encompases all generating group bounding spheres cumulatively (i.e. not
  // just bounding the children geometry).
  nvclusterlod_Sphere boundingSphere;

  // Cumulative maximum error in all children geometry due to mesh decimation
  float maxClusterQuadricError NVCLUSTERLOD_DEFAULT(0.0f);
} nvclusterlod_HierarchyNode;

NVCLUSTERLOD_STATIC_ASSERT(sizeof(nvclusterlod_HierarchyNode) == 4 + 16 + 4)

// Output data for a spatial hierarchy to accelerate LOD selection
typedef struct nvclusterlod_HierarchyOutput
{
  // Spatial hierarchy of bounding spheres. Nodes reference other nodes in this
  // array. There is actually a hierarchy per LOD level. Since LOD selection is
  // the typical use case, the roots of each are merged into a single tree for
  // convenience.
  nvclusterlod_HierarchyNode* nodes NVCLUSTERLOD_DEFAULT(nullptr);

  // Bounding sphere for each cluster group, encompassing all generating groups
  // within each group recursively
  nvclusterlod_Sphere* groupCumulativeBoundingSpheres NVCLUSTERLOD_DEFAULT(nullptr);

  // Quadric error of the group and the maximum quadric error of all generating
  // groups recursively
  float* groupCumulativeQuadricError NVCLUSTERLOD_DEFAULT(nullptr);

  // Number of nodes in the tree
  uint32_t nodeCount NVCLUSTERLOD_DEFAULT(0u);
} nvclusterlod_HierarchyOutput;

// Memory requirements to build a spatial hierarchy of LODs
typedef struct nvclusterlod_HierarchyCounts
{
  uint32_t nodeCount NVCLUSTERLOD_DEFAULT(0u);
} nvclusterlod_HierarchyCounts;

// Get the memory requirements for the spatial hierarchy output
nvclusterlod_Result nvclusterlodGetHierarchyRequirements(nvclusterlod_Context               context,
                                                         const nvclusterlod_HierarchyInput* input,
                                                         nvclusterlod_HierarchyCounts*      counts);

// Generate a spatial hierarchy of all LODs, merged into a single tree
nvclusterlod_Result nvclusterlodBuildHierarchy(nvclusterlod_Context               context,
                                               const nvclusterlod_HierarchyInput* input,
                                               nvclusterlod_HierarchyOutput*      output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NVCLUSTERLOD_HIERARCHY_H
