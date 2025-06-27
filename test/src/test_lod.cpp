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

#include <algorithm>
#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <nvclusterlod/nvclusterlod_cache.hpp>
#include <nvclusterlod/nvclusterlod_common.h>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <ostream>
#include <random>
#include <span>
#include <unordered_map>

#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif

// GLSL-like type definitions
using uvec3 = std::array<uint32_t, 3>;
using vec3  = std::array<float, 3>;
using vec4  = std::array<float, 4>;

vec3 getSpherePosition(const nvclusterlod_Sphere& sphere)
{
  return vec3{sphere.center.x, sphere.center.y, sphere.center.z};
}

// Adds two vec3s.
vec3 add(const vec3& a, const vec3& b)
{
  return vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

// Subtracts two vec3s.
vec3 sub(const vec3& a, const vec3& b)
{
  return vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Multiplies a vec3 by a constant value.
vec3 mul(const vec3& v, float a)
{
  return vec3{v[0] * a, v[1] * a, v[2] * a};
}

// Returns the squared length of a vec3.
float lengthSquared(const vec3& v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

// Returns the length of a vec3.
float length(const vec3& v)
{
  return std::sqrt(lengthSquared(v));
}

// Normalizes a vec3.
vec3 normalize(const vec3& v)
{
  const float vecLength = length(v);
  const float factor    = (vecLength == 0.0f) ? 1.0f : (1.0f / vecLength);
  return mul(v, factor);
}

// Define a hash for vec3, so that we can use it in std::unordered_map.
template <>
struct std::hash<vec3>
{
  std::size_t operator()(const vec3& v) const noexcept
  {
    // This doesn't need to be a good hash; it just needs to exist.
    const std::hash<float> hasher{};
    return hasher(v[0]) + 3 * hasher(v[1]) + 5 * hasher(v[2]);
  }
};


struct mat4
{
  std::array<vec4, 4> columns{};
  static mat4         identity() { return mat4{.columns = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}}}; }
  static mat4         makeTranslation(vec3 translation)
  {
    mat4 result = mat4::identity();
    for(size_t i = 0; i < 3; i++)
    {
      result.columns[3][i] = translation[i];
    }
    return result;
  }
};

vec3 transformPoint(const mat4& t, const vec3& point)
{
  vec3 result = {t.columns[3][0], t.columns[3][1], t.columns[3][2]};
  for(size_t i = 0; i < 3; i++)
  {
    for(size_t row = 0; row < 3; row++)
    {
      result[row] += t.columns[i][row] * point[i];
    }
  }
  return result;
}

// Random number generator.
std::default_random_engine rng(123);
// Returns a uniform random point on a sphere.
vec3 randomPointOnSphere(const nvclusterlod_Sphere& sphere)
{
  // From https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres

  // Random Z coordinate on a unit sphere, in the range [-1, 1].
  const float z = 1.F - 2.F * (static_cast<float>(rng()) / static_cast<float>(rng.max()));
  // Choose a random point on the surface of the sphere at this z coordinate:
  const float r                  = sqrtf(1.F - z * z);
  const float phi                = 2.0f * M_PIf * (static_cast<float>(rng()) / static_cast<float>(rng.max()));
  const vec3  randomOnUnitSphere = {r * cosf(phi), r * sinf(phi), z};
  // Now scale and translate this.
  return add(getSpherePosition(sphere), mul(randomOnUnitSphere, sphere.radius));
}

// Icosahedron data.
namespace icosahedron {
constexpr float              X         = .525731112119133606f;
constexpr float              Z         = .850650808352039932f;
static std::array<vec3, 12>  positions = {{{-X, 0.0, Z},
                                           {X, 0.0, Z},
                                           {-X, 0.0, -Z},
                                           {X, 0.0, -Z},
                                           {0.0, Z, X},
                                           {0.0, Z, -X},
                                           {0.0, -Z, X},
                                           {0.0, -Z, -X},
                                           {Z, X, 0.0},
                                           {-Z, X, 0.0},
                                           {Z, -X, 0.0},
                                           {-Z, -X, 0.0}}};
static std::array<uvec3, 20> triangles = {{{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
                                           {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
                                           {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                                           {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}}};
}  // namespace icosahedron

// Type of a function to call when creating a triangle. Takes 3 positions as
// inputs.
using triangle_callback = std::function<void(vec3, vec3, vec3)>;

// Recursively subdivides a triangle on a sphere by a factor of 2^depth.
// Calls the callback function on each new triangle.
void subdivide(vec3 v0, vec3 v1, vec3 v2, int depth, triangle_callback& callback)
{
  if(depth == 0)
  {
    callback(v0, v1, v2);
  }
  else
  {
    vec3 v01 = normalize(add(v0, v1));
    vec3 v12 = normalize(add(v1, v2));
    vec3 v20 = normalize(add(v2, v0));
    subdivide(v0, v01, v20, depth - 1, callback);
    subdivide(v1, v12, v01, depth - 1, callback);
    subdivide(v2, v20, v12, depth - 1, callback);
    subdivide(v01, v12, v20, depth - 1, callback);
  }
}

// Makes an icosphere with 20 * (4^depth) triangles.
void makeIcosphere(int depth, triangle_callback& callback)
{
  for(size_t i = 0; i < icosahedron::triangles.size(); i++)
  {
    const vec3 v0 = icosahedron::positions[icosahedron::triangles[i][0]];
    const vec3 v1 = icosahedron::positions[icosahedron::triangles[i][1]];
    const vec3 v2 = icosahedron::positions[icosahedron::triangles[i][2]];
    subdivide(v0, v1, v2, depth, callback);
  }
}

// Writes the geometry part of the Wavefront .obj format to a stream.
void writeObjGeometry(std::ostream& os, std::span<const uvec3> triangles, std::span<const vec3> positions)
{
  for(auto& p : positions)
  {
    os << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
  }

  for(auto& t : triangles)
  {
    os << "f " << t[0] + 1 << " " << t[1] + 1 << " " << t[2] + 1 << "\n";
  }
}

struct GeometryMesh
{
  std::vector<uvec3> triangles;
  std::vector<vec3>  positions;
  void               writeObj(const std::string& path)
  {
    std::ofstream f(path);
    f << "g mesh\n";
    writeObjGeometry(f, triangles, positions);
  }
};

GeometryMesh makeIcosphere(int subdivision)
{
  std::unordered_map<vec3, uint32_t> vertexCache;
  std::vector<uvec3>                 triangles;
  // Our triangle callback function tries to place each of the vertices in the
  // vertex cache; each of the `it` iterators point to the existing value if
  // the vertex was already in the cache, or to a new value at the end of the
  // cache if it's a new vertex.
  triangle_callback callback = [&vertexCache, &triangles](vec3 v0, vec3 v1, vec3 v2) {
    auto [it0, new0] = vertexCache.try_emplace(v0, static_cast<uint32_t>(vertexCache.size()));
    auto [it1, new1] = vertexCache.try_emplace(v1, static_cast<uint32_t>(vertexCache.size()));
    auto [it2, new2] = vertexCache.try_emplace(v2, static_cast<uint32_t>(vertexCache.size()));
    triangles.push_back({it0->second, it1->second, it2->second});
  };
  makeIcosphere(subdivision, callback);
  std::vector<vec3> positions(vertexCache.size());
  for(const auto& [position, index] : vertexCache)
  {
    positions[index] = position;
  }
  return GeometryMesh{triangles, positions};
}

// Computes the conservative maximum arcsine of any geometric error relative to
// the camera, where 'transform' defines a transformation to eye-space.
float conservativeErrorOverDistance(const mat4& transform, const nvclusterlod_Sphere& boundingSphere, float objectSpaceQuadricError)
{
  float radiusScale    = 1.0f;
  float maxError       = objectSpaceQuadricError * radiusScale;
  float sphereDistance = length(transformPoint(transform, getSpherePosition(boundingSphere)));
  float errorDistance  = std::max(maxError, sphereDistance - boundingSphere.radius * radiusScale);
  return maxError / errorDistance;
}

bool traverseChild(const mat4& viewInstanceTransform, const nvclusterlod_HierarchyNode& node, float errorOverDistanceThreshold)
{
  return conservativeErrorOverDistance(viewInstanceTransform, node.boundingSphere, node.maxClusterQuadricError) >= errorOverDistanceThreshold;
}

bool renderCluster(const mat4& viewInstanceTransform, float quadricError, const nvclusterlod_Sphere& boundingSphere, float errorOverDistanceThreshold)
{
  return conservativeErrorOverDistance(viewInstanceTransform, boundingSphere, quadricError) < errorOverDistanceThreshold;
}

bool traverseChild(const vec3& cameraPosition, const nvclusterlod_HierarchyNode& node, float errorOverDistanceThreshold)
{
  return traverseChild(mat4::makeTranslation(mul(cameraPosition, -1.f)), node, errorOverDistanceThreshold);
}

bool renderCluster(const vec3& cameraPosition, float quadricError, const nvclusterlod_Sphere& boundingSphere, float errorOverDistanceThreshold)
{
  return renderCluster(mat4::makeTranslation(mul(cameraPosition, -1.f)), quadricError, boundingSphere, errorOverDistanceThreshold);
}

// Returns whether `inner` is inside or equal to `outer`.
bool isInside(const nvclusterlod_Sphere& inner, const nvclusterlod_Sphere& outer)
{
  const float radiusDifference = outer.radius - inner.radius;
  return (radiusDifference >= 0.0f)  // if this is negative then `inner` cannot be inside `outer`
         && lengthSquared(sub(getSpherePosition(inner), getSpherePosition(outer))) <= radiusDifference * radiusDifference;
}

// Verifies that for this node:
// - if it is a leaf node, each cluster's bounding sphere is contained within
//   the node's bounding sphere.
// - if it is an internal node, each child's bounding sphere is contained
//   within the node's bounding sphere.
void verifyNodeRecursive(const nvclusterlod::LocalizedLodMesh& m, const nvclusterlod::LodHierarchy& h, const nvclusterlod_HierarchyNode& node)
{
  if(node.clusterGroup.isClusterGroup)
  {
    // Verify real cluster bounding spheres
    // TODO: cluster bounding spheres in hierarchy are not bounding spheres of
    // the cluster, but bounding spheres of the cluster's generating group.
    const nvcluster_Range& clusterRange = m.lodMesh.groupClusterRanges[node.clusterGroup.group];
    for(uint32_t i = 0; i < clusterRange.count; i++)
    {
      const nvclusterlod_Sphere& clusterSphere = m.lodMesh.clusterBoundingSpheres[clusterRange.offset + i];
#if 0  // For debugging
          printf("A = Sphere((%f, %f, %f), %f)\n", node.boundingSphere.x, node.boundingSphere.y, node.boundingSphere.z, node.boundingSphere.radius);
          printf("B = Sphere((%f, %f, %f), %f)\n", clusterSphere.x, clusterSphere.y, clusterSphere.z, clusterSphere.radius);
          assert(isInside(clusterSphere, node.boundingSphere));
#endif
      EXPECT_TRUE(isInside(clusterSphere, node.boundingSphere));
    }
  }
  else
  {
    for(uint32_t i = 0; i <= node.children.childCountMinusOne; i++)
    {
      const nvclusterlod_HierarchyNode& child = h.nodes[node.children.childOffset + i];
      assert(isInside(child.boundingSphere, node.boundingSphere));
      EXPECT_TRUE(isInside(child.boundingSphere, node.boundingSphere));
      verifyNodeRecursive(m, h, child);
    }
  }
}

inline nvclusterlod_Sphere generatingSphere(std::span<const nvclusterlod_Sphere> groupSpheres, uint32_t generatingGroupIndex)
{
  return (generatingGroupIndex == NVCLUSTERLOD_ORIGINAL_MESH_GROUP) ? nvclusterlod_Sphere{} : groupSpheres[generatingGroupIndex];
}

inline float generatingError(std::span<const float> groupErrors, uint32_t generatingGroupIndex)
{
  return (generatingGroupIndex == NVCLUSTERLOD_ORIGINAL_MESH_GROUP) ? 0.0f : groupErrors[generatingGroupIndex];
}

// Checks that there can be no overlapping geometry given two clusters from
// different LOD levels that represent the same surfcae. See renderCluster().
void verifyMutuallyExclusive(const nvclusterlod::LocalizedLodMesh& m,
                             const nvclusterlod::LodHierarchy&     h,
                             uint32_t                              cluster0,
                             const nvclusterlod_HierarchyNode&     node0,
                             uint32_t                              cluster1,
                             const nvclusterlod_HierarchyNode&     node1)
{
  const nvclusterlod_Sphere cluster0Sphere =
      generatingSphere(h.groupCumulativeBoundingSpheres, m.lodMesh.clusterGeneratingGroups[cluster0]);
  const nvclusterlod_Sphere cluster1Sphere =
      generatingSphere(h.groupCumulativeBoundingSpheres, m.lodMesh.clusterGeneratingGroups[cluster1]);
  const float cluster0QuadricError = generatingError(h.groupCumulativeQuadricError, m.lodMesh.clusterGeneratingGroups[cluster0]);
  const float cluster1QuadricError = generatingError(h.groupCumulativeQuadricError, m.lodMesh.clusterGeneratingGroups[cluster1]);

  const float errorOverDistanceThreshold = 0.9999f;  // near worst case

  for(int i = 0; i < 10; ++i)
  {
    const vec3 testCameraPos = randomPointOnSphere(node0.boundingSphere);
    const bool begin0        = traverseChild(testCameraPos, node0, errorOverDistanceThreshold);
    const bool begin1        = traverseChild(testCameraPos, node1, errorOverDistanceThreshold);
    bool       end0 = !renderCluster(testCameraPos, cluster0QuadricError, cluster0Sphere, errorOverDistanceThreshold);
    bool       end1 = !renderCluster(testCameraPos, cluster1QuadricError, cluster1Sphere, errorOverDistanceThreshold);
    bool       bothVisible = begin0 && !end0 && begin1 && !end1;
    EXPECT_FALSE(bothVisible);
  }
}

// Write a mesh and a generated LOD mesh. Useful for debugging.
void writeDebugLodMesh(const std::string& filename, const GeometryMesh& mesh, const nvclusterlod::LodMesh& lodMesh)
{
  std::ofstream f(filename);
  writeObjGeometry(f, {}, mesh.positions);
  for(size_t clusterIndex = 0; clusterIndex < lodMesh.clusterTriangleRanges.size(); ++clusterIndex)
  {
    f << "o cluster" << clusterIndex << "\n";
    f << "g cluster" << clusterIndex << "\n";
    const nvcluster_Range& triangleRange = lodMesh.clusterTriangleRanges[clusterIndex];
    const uvec3* pTriangleVertices = reinterpret_cast<const uvec3*>(lodMesh.triangleVertices.data()) + triangleRange.offset;
    writeObjGeometry(f, {pTriangleVertices, triangleRange.count}, {});
  }
}

// This nvcluster_Context cleans itself up so that we don't leak memory if
// a test exits early.
struct ScopedContext
{
  nvcluster_Context context = nullptr;

  nvcluster_Result init()
  {
    assert(!context);
    nvcluster_ContextCreateInfo info{};
    return nvclusterCreateContext(&info, &context);
  }

  ~ScopedContext() { std::ignore = nvclusterDestroyContext(context); }
};

// This nvclusterlod_Context cleans itself up so that we don't leak memory if
// a test exits early.
struct ScopedLodContext
{
  nvclusterlod_Context context = nullptr;

  nvclusterlod_Result init(nvcluster_Context baseContext)
  {
    assert(!context);
    nvclusterlod_ContextCreateInfo info{.clusterContext = baseContext};
    return nvclusterlodCreateContext(&info, &context);
  }

  ~ScopedLodContext() { std::ignore = nvclusterlodDestroyContext(context); }
};


// GoogleTest does not offer any nice way to share data between tests. Since the
// test data takes some time to generate, everything is just in one giant test.
TEST(Hierarchy, BoundsAndOverlaps)
{
  ScopedContext context;
  ASSERT_EQ(context.init(), nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  ScopedLodContext lodContext;
  ASSERT_EQ(lodContext.init(context.context), nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);

  GeometryMesh           icosphere{makeIcosphere(6)};
  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3),
      .clusterConfig =
          {
              .minClusterSize    = 32,  // force clusters of 32 triangles
              .maxClusterSize    = 32,
              .costUnderfill     = 0.9f,
              .costOverlap       = 0.5f,
              .preSplitThreshold = 1u << 17,
          },
      .groupConfig =
          {
              .minClusterSize    = 32,  // force groups of 32 clusters
              .maxClusterSize    = 32,
              .costUnderfill     = 0.5f,
              .costOverlap       = 0.0f,
              .preSplitThreshold = 0,
          },
      .decimationFactor = 0.5f,
  };

  nvclusterlod::LocalizedLodMesh mesh;
  nvclusterlod_Result            result = nvclusterlod::generateLocalizedLodMesh(lodContext.context, meshInput, mesh);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  EXPECT_EQ(mesh.maxClusterTriangles, 32);
  EXPECT_GE(mesh.maxClusterVertices, 32 / 2 + 2);  // 32 triangle polyhedron
  EXPECT_LE(mesh.maxClusterVertices, 32 * 3);      // 32 unconnected triangles

  const nvclusterlod_HierarchyInput hierarchyInput{
      .clusterGeneratingGroups = mesh.lodMesh.clusterGeneratingGroups.data(),
      .clusterBoundingSpheres  = mesh.lodMesh.clusterBoundingSpheres.data(),
      .groupQuadricErrors      = mesh.lodMesh.groupQuadricErrors.data(),
      .groupClusterRanges      = mesh.lodMesh.groupClusterRanges.data(),
      .lodLevelGroupRanges     = mesh.lodMesh.lodLevelGroupRanges.data(),
      .clusterCount            = static_cast<uint32_t>(mesh.lodMesh.clusterBoundingSpheres.size()),
      .groupCount              = static_cast<uint32_t>(mesh.lodMesh.groupClusterRanges.size()),
      .lodLevelCount           = static_cast<uint32_t>(mesh.lodMesh.lodLevelGroupRanges.size()),
  };
  nvclusterlod::LodHierarchy hierarchy;
  result = nvclusterlod::generateLodHierarchy(lodContext.context, hierarchyInput, hierarchy);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);

  verifyNodeRecursive(mesh, hierarchy, hierarchy.nodes[0]);

  // Build sets of generating groups that contributed clusters for decimation
  // into each group.
  nvclusterlod::GroupGeneratingGroups groupGeneratingGroups;
  result = nvclusterlod::generateGroupGeneratingGroups(mesh.lodMesh.groupClusterRanges,
                                                       mesh.lodMesh.clusterGeneratingGroups, groupGeneratingGroups);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);

  std::vector<uint32_t> clusterNodes(mesh.lodMesh.clusterTriangleRanges.size());
  for(size_t nodeIndex = 0; nodeIndex < hierarchy.nodes.size(); ++nodeIndex)
  {
    if(hierarchy.nodes[nodeIndex].clusterGroup.isClusterGroup)
    {
      const nvcluster_Range& clusterIndices = mesh.lodMesh.groupClusterRanges[hierarchy.nodes[nodeIndex].clusterGroup.group];
      for(uint32_t i = clusterIndices.offset; i < clusterIndices.offset + clusterIndices.count; i++)
      {
        clusterNodes[i] = uint32_t(nodeIndex);
      }
    }
  }

  // Verify that each cluster's generating group's generating groups do not have
  // any clusters that can be drawn at the same time. Reverse order as the last
  // few low-detail clusters are where the interesting cases are.

  // Loop from mesh.lodMesh.groupClusterRanges.size() - 1 to 0.
  for(size_t groupIndex = mesh.lodMesh.groupClusterRanges.size(); (groupIndex--) > 0;)
  {
    const nvcluster_Range groupClusterRange = mesh.lodMesh.groupClusterRanges[groupIndex];
    for(uint32_t clusterIndex = groupClusterRange.offset;
        clusterIndex < groupClusterRange.offset + groupClusterRange.count; clusterIndex++)
    {
      const uint32_t generatingGroupIndex = mesh.lodMesh.clusterGeneratingGroups[clusterIndex];
      if(generatingGroupIndex == NVCLUSTERLOD_ORIGINAL_MESH_GROUP)
        continue;
      for(uint32_t generatingGeneratingGroupIndex : groupGeneratingGroups[generatingGroupIndex])
      {
        uint32_t               lastNode = 0xffffffff;  // Initialize to some value that doesn't appear in clusterNodes
        const nvcluster_Range generatingGeneratingClusterRange = mesh.lodMesh.groupClusterRanges[generatingGeneratingGroupIndex];
        for(uint32_t i = 0; i < generatingGeneratingClusterRange.count; i++)
        {
          const uint32_t generatingGeneratingClusterIndex = generatingGeneratingClusterRange.offset + i;
          // Skip clusters with the same nodes. The child node is a primary
          // interest. Faster testing at the expense of a few less test cases that
          // are near identical.
          if(clusterNodes[generatingGeneratingClusterIndex] == lastNode)
            continue;
          lastNode = clusterNodes[generatingGeneratingClusterIndex];

          verifyMutuallyExclusive(mesh, hierarchy, clusterIndex, hierarchy.nodes[clusterNodes[clusterIndex]],
                                  generatingGeneratingClusterIndex,
                                  hierarchy.nodes[clusterNodes[generatingGeneratingClusterIndex]]);
        }
      }
    }
  }
};

TEST(Mesh, VertexLimit)
{
  ScopedContext context;
  ASSERT_EQ(context.init(), nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  ScopedLodContext lodContext;
  ASSERT_EQ(lodContext.init(context.context), nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);

  GeometryMesh           icosphere{makeIcosphere(4)};
  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3),
      .clusterConfig =
          {
              .minClusterSize        = 1,
              .maxClusterSize        = 1000,
              .maxClusterVertices    = 32,
              .costUnderfill         = 0.9f,
              .costOverlap           = 0.5f,
              .costUnderfillVertices = 0.9f,
              .preSplitThreshold     = 1u << 17,
          },
      .groupConfig =
          {
              .minClusterSize    = 32,
              .maxClusterSize    = 32,
              .costUnderfill     = 0.5f,
              .costOverlap       = 0.0f,
              .preSplitThreshold = 0,
          },
      .decimationFactor = 0.5f,
  };

  nvclusterlod::LocalizedLodMesh mesh;
  nvclusterlod_Result            result = nvclusterlod::generateLocalizedLodMesh(lodContext.context, meshInput, mesh);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  EXPECT_LE(mesh.maxClusterVertices, meshInput.clusterConfig.maxClusterVertices);
  for(size_t i = 0; i < mesh.lodMesh.clusterTriangleRanges.size(); ++i)
  {
    const nvcluster_Range& triangleRange = mesh.lodMesh.clusterTriangleRanges[i];
    EXPECT_LE(triangleRange.count, mesh.maxClusterTriangles);
    std::unordered_set<uint32_t> uniqueVertices;
    for(size_t j = 0; j < triangleRange.count; ++j)
    {
      const nvclusterlod_Vec3u& vertex = mesh.lodMesh.triangleVertices[triangleRange.offset + j];
      uniqueVertices.insert(vertex.x);
      uniqueVertices.insert(vertex.y);
      uniqueVertices.insert(vertex.z);
    }
    EXPECT_LE(uniqueVertices.size(), mesh.maxClusterVertices);
  }
}

TEST(Hierarchy, AngularError)
{
  {
    // Test with some hard coded ballpark-real values
    const float fov  = M_PIf * 0.5f;
    const float qeod = nvclusterlodErrorOverDistance(1.0f, fov, 2048.0f);
    EXPECT_NEAR(qeod, 0.0004f, 0.0001f);
    EXPECT_NEAR(nvclusterlodPixelError(qeod, fov, 2048.0f), 1.0f, 1e-10f);
  }

  // Verify the reverse mapping works for many values
  std::array<float, 4> fovValues   = {M_PIf * 0.05f, M_PIf * 0.3f, M_PIf * 0.5f, M_PIf * 0.7f};
  std::array<float, 5> pixelValues = {0.5f, 1.0f, 10.0f, 100.0f, 1000.0f};
  for(float fov : fovValues)
  {
    for(float errorSize : pixelValues)
    {
      // TODO: would be nice if the precision were a bit better even for the
      // more extreme values
      float qeod = nvclusterlodErrorOverDistance(errorSize, fov, 2048.0f);
      EXPECT_NEAR(nvclusterlodPixelError(qeod, fov, 2048.0f), errorSize, 1e-4f);
    }
  }
}

extern "C" int runCTest(void);

TEST(CAPI, Contexts)
{
  EXPECT_EQ(runCTest(), 1);
}
