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
#include <nvcluster/util/objects.hpp>
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

#if TESTS_HAVE_MESHOPTIMIZER
#include <meshoptimizer.h>
#endif

using nvcluster::vec3f;
using nvcluster::vec3u;
using nvcluster::vec4f;

#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif

struct Sphere
{
  vec3f center;
  float radius;
  bool  operator==(const Sphere& other) const { return center == other.center && radius == other.radius; }
  operator nvclusterlod_Sphere() const { return {center, radius}; }
};

struct mat4
{
  std::array<vec4f, 4> columns{};
  static mat4 identity() { return mat4{.columns = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}}}; }
  static mat4 makeTranslation(vec3f translation)
  {
    mat4 result = mat4::identity();
    for(size_t i = 0; i < 3; i++)
    {
      result.columns[3][i] = translation[i];
    }
    return result;
  }
};

vec3f transformPoint(const mat4& t, const vec3f& point)
{
  vec3f result = {t.columns[3][0], t.columns[3][1], t.columns[3][2]};
  for(size_t i = 0; i < 3; i++)
  {
    for(size_t row = 0; row < 3; row++)
    {
      result[row] += t.columns[i][row] * point[i];
    }
  }
  return result;
}

// Returns a uniform random point on a sphere.
vec3f randomPointOnSphere(const Sphere& sphere)
{
  // Random number generator
  static std::default_random_engine rng(123);  // not thread safe

  // From https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres

  // Random Z coordinate on a unit sphere, in the range [-1, 1].
  const float z = 1.F - 2.F * (static_cast<float>(rng()) / static_cast<float>(rng.max()));
  // Choose a random point on the surface of the sphere at this z coordinate:
  const float r                  = sqrtf(1.F - z * z);
  const float phi                = 2.0f * M_PIf * (static_cast<float>(rng()) / static_cast<float>(rng.max()));
  const vec3f randomOnUnitSphere = {r * cosf(phi), r * sinf(phi), z};
  // Now scale and translate this.
  return sphere.center + randomOnUnitSphere * sphere.radius;
}

// Icosahedron data.
namespace icosahedron {
constexpr float              X         = .525731112119133606f;
constexpr float              Z         = .850650808352039932f;
static std::array<vec3f, 12> positions = {{{-X, 0.0, Z},
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
static std::array<vec3u, 20> triangles = {{{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
                                           {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
                                           {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                                           {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}}};
}  // namespace icosahedron

// Type of a function to call when creating a triangle. Takes 3 positions as
// inputs.
using triangle_callback = std::function<void(vec3f, vec3f, vec3f)>;

// Recursively subdivides a triangle on a sphere by a factor of 2^depth.
// Calls the callback function on each new triangle.
void subdivide(vec3f v0, vec3f v1, vec3f v2, int depth, triangle_callback& callback)
{
  if(depth == 0)
  {
    callback(v0, v1, v2);
  }
  else
  {
    vec3f v01 = normalize(v0 + v1);
    vec3f v12 = normalize(v1 + v2);
    vec3f v20 = normalize(v2 + v0);
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
    const vec3f v0 = icosahedron::positions[icosahedron::triangles[i][0]];
    const vec3f v1 = icosahedron::positions[icosahedron::triangles[i][1]];
    const vec3f v2 = icosahedron::positions[icosahedron::triangles[i][2]];
    subdivide(v0, v1, v2, depth, callback);
  }
}

// Writes the geometry part of the Wavefront .obj format to a stream.
void writeObjGeometry(std::ostream& os, std::span<const vec3u> triangles, std::span<const vec3f> positions)
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
  std::vector<vec3u> triangles;
  std::vector<vec3f> positions;
  void               writeObj(const std::string& path)
  {
    std::ofstream f(path);
    f << "g mesh\n";
    writeObjGeometry(f, triangles, positions);
  }
};

GeometryMesh makeIcosphere(int subdivision)
{
  std::unordered_map<vec3f, uint32_t> vertexCache;
  std::vector<vec3u>                  triangles;
  // Our triangle callback function tries to place each of the vertices in the
  // vertex cache; each of the `it` iterators point to the existing value if
  // the vertex was already in the cache, or to a new value at the end of the
  // cache if it's a new vertex.
  triangle_callback callback = [&vertexCache, &triangles](vec3f v0, vec3f v1, vec3f v2) {
    auto [it0, new0] = vertexCache.try_emplace(v0, static_cast<uint32_t>(vertexCache.size()));
    auto [it1, new1] = vertexCache.try_emplace(v1, static_cast<uint32_t>(vertexCache.size()));
    auto [it2, new2] = vertexCache.try_emplace(v2, static_cast<uint32_t>(vertexCache.size()));
    triangles.push_back({it0->second, it1->second, it2->second});
  };
  makeIcosphere(subdivision, callback);
  std::vector<vec3f> positions(vertexCache.size());
  for(const auto& [position, index] : vertexCache)
  {
    positions[index] = position;
  }
  return GeometryMesh{triangles, positions};
}

// Collapses edges until the output triangle count is met. Prioritizes
// collapsing edges with at least one non-locked vertex. Returns the number of
// triangles written to collapsedTriangles.
uint32_t collapseUnlockedEdges(std::span<const vec3u>   triangles,
                               std::span<const uint8_t> lockedVertices,
                               std::span<vec3u>         collapsedTriangles,
                               uint32_t                 targetTriangleCount)
{
  assert(!collapsedTriangles.empty());

  // Duplicate the input triangles as we need to modify them
  std::ranges::copy(triangles, collapsedTriangles.begin());

  // Track the active set of triangles that have not been made degenerate as
  // part of an edge collapse. This allows indices into collapsedTriangles to
  // remain valid.
  std::set<uint32_t> remainingTriangles;
  for(uint32_t i = 0; i < uint32_t(triangles.size()); i++)
  {
    remainingTriangles.insert(i);
  }

  // Maintain a list of triangles per vertex for quick lookup
  std::unordered_map<uint32_t, std::vector<uint32_t>> vertexTriangles;
  for(uint32_t i = 0; i < triangles.size(); i++)
  {
    for(uint32_t j = 0; j < 3; j++)
    {
      vertexTriangles[triangles[i][j]].push_back(i);
    }
  }

  // Collapse edges until the output triangle count is met
  std::mt19937 g(0);
  while(remainingTriangles.size() > targetTriangleCount)
  {
    // Find the first non-locked edge, {maybe-locked vertex, not-locked vertex}
    std::optional<std::pair<uint32_t, uint32_t>> edge;
    std::vector<uint32_t> shuffledRemainingTriangles(remainingTriangles.begin(), remainingTriangles.end());  // wasteful but quick
    std::shuffle(shuffledRemainingTriangles.begin(), shuffledRemainingTriangles.end(), g);
    for(const uint32_t triangleIndex : shuffledRemainingTriangles)
    {
      const vec3u& triangle = collapsedTriangles[triangleIndex];
      for(uint32_t i = 0; i < 3 && !edge; i++)
      {
        if(lockedVertices[triangle[i]] == 0 && lockedVertices[triangle[(i + 2) % 3]] == 0)
          edge = std::make_pair(triangle[(i + 1) % 3], triangle[i]);
        else if(lockedVertices[triangle[(i + 1) % 3]] == 0 && lockedVertices[triangle[(i + 2) % 3]] == 0)
          edge = std::make_pair(triangle[i], triangle[(i + 1) % 3]);
      }
      if(edge)
      {
        break;
      }
    }

    // If no non-locked edge was found, start collapsing arbitrary edges
    if(!edge)
      edge = std::make_pair(collapsedTriangles[shuffledRemainingTriangles.front()][0],
                            collapsedTriangles[shuffledRemainingTriangles.front()][1]);

    // Collapse the edge (second vertex becomes the first vertex)
    assert(vertexTriangles.find(edge->second) != vertexTriangles.end());
    assert(edge->second != edge->first);  // doesn't handle degenerates in input
    for(uint32_t triangleIndex : vertexTriangles[edge->second])
    {
      assert(triangleIndex < uint32_t(collapsedTriangles.size()));
      vec3u& triangle = collapsedTriangles[triangleIndex];
      for(uint32_t i = 0; i < 3; i++)
      {
        if(triangle[i] == edge->second)
        {
          if(triangle[(i + 1) % 3] == edge->first || triangle[(i + 2) % 3] == edge->first)
          {
            // Remove triangles that include the edge as they become degenerate
            remainingTriangles.erase(triangleIndex);
            break;
          }
          triangle[i] = edge->first;
          vertexTriangles[edge->first].push_back(triangleIndex);
        }
      }
    }
  }

  // Compact the sparse result. This requires remainingTriangles to be in
  // ascending order.
  static_assert(std::same_as<decltype(remainingTriangles), std::set<uint32_t>>);
  uint32_t collapsedTriangleCount = 0;
  for(uint32_t triangleIndex : remainingTriangles)
  {
    collapsedTriangles[collapsedTriangleCount++] = collapsedTriangles[triangleIndex];
  }
  return collapsedTriangleCount;
}

nvcluster_Bool decimateTrianglesFallback(void* /* userData */,
                                         const nvclusterlod_DecimateTrianglesCallbackParams* params,
                                         nvclusterlod_DecimateTrianglesCallbackResult*       result)
{
  uint32_t collapsedTriangleCount =
      collapseUnlockedEdges(std::span{reinterpret_cast<const vec3u*>(params->triangleVertices), params->triangleCount},
                            std::span{params->vertexLockFlags, params->vertexCount},
                            std::span{reinterpret_cast<vec3u*>(params->decimatedTriangleVertices), params->triangleCount},
                            params->targetTriangleCount);
  *result = nvclusterlod_DecimateTrianglesCallbackResult{
      .decimatedTriangleCount = collapsedTriangleCount, .additionalVertexCount = 0u, .quadricError = 1.0f};
  return NVCLUSTER_TRUE;
}

// Computes the conservative maximum arcsine of any geometric error relative to
// the camera, where 'transform' defines a transformation to eye-space.
float conservativeErrorOverDistance(const mat4& transform, const nvclusterlod_Sphere& boundingSphereC, float objectSpaceQuadricError)
{
  auto  boundingSphere = std::bit_cast<Sphere>(boundingSphereC);
  float radiusScale    = 1.0f;
  float maxError       = objectSpaceQuadricError * radiusScale;
  float sphereDistance = length(transformPoint(transform, boundingSphere.center));
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

bool traverseChild(const vec3f& cameraPosition, const nvclusterlod_HierarchyNode& node, float errorOverDistanceThreshold)
{
  return traverseChild(mat4::makeTranslation(-cameraPosition), node, errorOverDistanceThreshold);
}

bool renderCluster(const vec3f& cameraPosition, float quadricError, const nvclusterlod_Sphere& boundingSphere, float errorOverDistanceThreshold)
{
  return renderCluster(mat4::makeTranslation(-cameraPosition), quadricError, boundingSphere, errorOverDistanceThreshold);
}

// Returns whether `inner` is inside or equal to `outer`.
bool isInside(const Sphere& inner, const Sphere& outer)
{
  const float radiusDifference = outer.radius - inner.radius;
  return (radiusDifference >= 0.0f)  // if this is negative then `inner` cannot be inside `outer`
         && length_squared(inner.center - outer.center) <= radiusDifference * radiusDifference;
}

bool isInside(const nvclusterlod_Sphere& inner, const nvclusterlod_Sphere& outer)
{
  auto innerSphere = std::bit_cast<Sphere>(inner);
  auto outerSphere = std::bit_cast<Sphere>(outer);
  return isInside(innerSphere, outerSphere);
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
      const nvclusterlod_Sphere& clusterOriginalSphere = m.lodMesh.clusterBoundingSpheres[clusterRange.offset + i];
      const nvclusterlod_Sphere& clusterGroupSphere    = h.groupCumulativeBoundingSpheres[node.clusterGroup.group];

      // Logically, the decimated group's bounding sphere should be inside the
      // cumulative cluster bounding spheres. However, it is not included in the
      // cumulative sphere computation (see buildHierarchy()) and is not
      // guaranteed to be tight. If a decimated group's bounding sphere is
      // slightly oversized, it can poke outside of the bounding sphere of the
      // generating groups.
      auto biggerNodeSphere = node.boundingSphere;
      biggerNodeSphere.radius *= TESTS_HAVE_MESHOPTIMIZER ? 1.2f : 1.5f;
      EXPECT_TRUE(isInside(clusterOriginalSphere, biggerNodeSphere));
#if 0  // For debugging
      printf("A = Sphere((%f, %f, %f), %f)\n", node.boundingSphere.center.x, node.boundingSphere.center.y,
              node.boundingSphere.center.z, node.boundingSphere.radius);
      printf("B = Sphere((%f, %f, %f), %f)\n", clusterOriginalSphere.center.x, clusterOriginalSphere.center.y, clusterOriginalSphere.center.z,
        clusterOriginalSphere.radius);
      assert(isInside(clusterOriginalSphere, node.boundingSphere));
#endif

      // All but the per-LOD level root node bounding spheres should come from
      // the group cumulative bounding spheres
      if(node.boundingSphere.radius != std::numeric_limits<float>::max())
      {
        EXPECT_EQ(std::bit_cast<Sphere>(node.boundingSphere), std::bit_cast<Sphere>(clusterGroupSphere));
      }

      // Cluster group cumulative bounding spheres should enclose the generating
      // group spheres
      uint32_t clusterGeneratingGroupIndex = m.lodMesh.clusterGeneratingGroups[clusterRange.offset + i];
      if(clusterGeneratingGroupIndex != NVCLUSTERLOD_ORIGINAL_MESH_GROUP)
      {
        const nvclusterlod_Sphere& clusterGeneratingGroupSphere = h.groupCumulativeBoundingSpheres[clusterGeneratingGroupIndex];
        EXPECT_TRUE(isInside(clusterGeneratingGroupSphere, node.boundingSphere));
      }
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
    const vec3f testCameraPos = randomPointOnSphere(std::bit_cast<Sphere>(node0.boundingSphere));
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
    const vec3u* pTriangleVertices = reinterpret_cast<const vec3u*>(lodMesh.triangleVertices.data()) + triangleRange.offset;
    writeObjGeometry(f, {pTriangleVertices, triangleRange.count}, {});
  }
}

inline void check(nvcluster_Result result)
{
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
    throw std::runtime_error(nvclusterResultString(result));
}

inline void check(nvclusterlod_Result result)
{
  if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    throw std::runtime_error(nvclusterlodResultString(result));
}

// nvcluster_Context wrapper handles ownership, lifetime, doesn't leak when
// tests return etc.
struct ScopedContext
{
  ScopedContext(const nvcluster_ContextCreateInfo& createInfo = {})
  {
    check(nvclusterCreateContext(&createInfo, &context));
  }
  ~ScopedContext() { std::ignore = nvclusterDestroyContext(context); }
  ScopedContext(const ScopedContext& other)            = delete;
  ScopedContext& operator=(const ScopedContext& other) = delete;
  operator nvcluster_Context() const { return context; }
  nvcluster_Context context = nullptr;
};

// nvclusterlod_Context wrapper handles ownership, lifetime, doesn't leak when
// tests return etc.
struct ScopedLodContext
{
  ScopedLodContext(nvcluster_Context baseContext)
  {
    nvclusterlod_ContextCreateInfo info{.clusterContext = baseContext};
    check(nvclusterlodCreateContext(&info, &context));
  }
  ~ScopedLodContext() { std::ignore = nvclusterlodDestroyContext(context); }
  ScopedLodContext(const ScopedLodContext& other)            = delete;
  ScopedLodContext& operator=(const ScopedLodContext& other) = delete;
  operator nvclusterlod_Context() const { return context; }
  nvclusterlod_Context context = nullptr;
};

// GoogleTest does not offer any nice way to share data between tests. Since the
// test data takes some time to generate, everything is just in one giant test.
TEST(Hierarchy, BoundsAndOverlaps)
{
  ScopedContext          context;
  ScopedLodContext       lodContext(context);
  GeometryMesh           icosphere{makeIcosphere(6)};
  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3f),
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
      .decimationFactor          = 0.5f,
      .decimateTrianglesCallback = TESTS_HAVE_MESHOPTIMIZER ? nullptr : decimateTrianglesFallback,
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
  ScopedContext          context;
  ScopedLodContext       lodContext(context);
  GeometryMesh           icosphere{makeIcosphere(4)};
  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3f),
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
      .decimationFactor          = 0.5f,
      .decimateTrianglesCallback = TESTS_HAVE_MESHOPTIMIZER ? nullptr : decimateTrianglesFallback,
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

TEST(Mesh, ClusterBoundingSpheresOptional)
{
  ScopedContext          context;
  ScopedLodContext       lodContext(context);
  GeometryMesh           icosphere{makeIcosphere(2)};
  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3f),
      .clusterConfig =
          {
              .minClusterSize    = 4,
              .maxClusterSize    = 32,
              .costUnderfill     = 0.9f,
              .costOverlap       = 0.5f,
              .preSplitThreshold = 1u << 17,
          },
      .groupConfig =
          {
              .minClusterSize    = 8,
              .maxClusterSize    = 16,
              .costUnderfill     = 0.5f,
              .costOverlap       = 0.0f,
              .preSplitThreshold = 0,
          },
      .decimationFactor          = 0.5f,
      .decimateTrianglesCallback = TESTS_HAVE_MESHOPTIMIZER ? nullptr : decimateTrianglesFallback,
  };

  nvclusterlod_MeshCounts counts{};
  ASSERT_EQ(nvclusterlodGetMeshRequirements(lodContext.context, &meshInput, &counts), nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);

  std::vector<nvclusterlod_Vec3u> triangleVertices(counts.triangleCount);
  std::vector<nvcluster_Range>    clusterTriangleRanges(counts.clusterCount);
  std::vector<uint32_t>           clusterGeneratingGroups(counts.clusterCount);
  std::vector<float>              groupQuadricErrors(counts.groupCount);
  std::vector<nvcluster_Range>    groupClusterRanges(counts.groupCount);
  std::vector<nvcluster_Range>    lodLevelGroupRanges(counts.lodLevelCount);

  nvclusterlod_MeshOutput meshOutput{};
  meshOutput.triangleVertices        = triangleVertices.data();
  meshOutput.clusterTriangleRanges   = clusterTriangleRanges.data();
  meshOutput.clusterGeneratingGroups = clusterGeneratingGroups.data();
  meshOutput.clusterBoundingSpheres  = nullptr;  // optional output: leave null
  meshOutput.groupQuadricErrors      = groupQuadricErrors.data();
  meshOutput.groupClusterRanges      = groupClusterRanges.data();
  meshOutput.lodLevelGroupRanges     = lodLevelGroupRanges.data();
  meshOutput.triangleCount           = static_cast<uint32_t>(triangleVertices.size());
  meshOutput.clusterCount            = static_cast<uint32_t>(clusterTriangleRanges.size());
  meshOutput.groupCount              = static_cast<uint32_t>(groupClusterRanges.size());
  meshOutput.lodLevelCount           = static_cast<uint32_t>(lodLevelGroupRanges.size());

  nvclusterlod_Result result = nvclusterlodBuildMesh(lodContext.context, &meshInput, &meshOutput);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  EXPECT_GT(meshOutput.clusterCount, 0u);
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
  auto fovValues   = std::to_array({M_PIf * 0.05f, M_PIf * 0.3f, M_PIf * 0.5f, M_PIf * 0.7f});
  auto pixelValues = std::to_array({0.5f, 1.0f, 10.0f, 100.0f, 1000.0f});
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

TEST(Mesh, DecimateTrianglesCallback)
{
  ScopedContext    context;
  ScopedLodContext lodContext(context);
  GeometryMesh     icosphere{makeIcosphere(2)};

  // Provide a decimation callback that uses meshopt_simplifySloppy
  std::atomic<bool> invoked             = false;  // the callback must be thread safe
  auto              decimateWithCapture = [&invoked](const nvclusterlod_DecimateTrianglesCallbackParams* params,
                                        nvclusterlod_DecimateTrianglesCallbackResult*       result) -> nvcluster_Bool {
    float quadricError = 0.0f;
#if TESTS_HAVE_MESHOPTIMIZER
    size_t simplifiedTriangleCount =
        meshopt_simplifySloppy(reinterpret_cast<unsigned int*>(params->decimatedTriangleVertices),
                               reinterpret_cast<const unsigned int*>(params->triangleVertices),
                               params->triangleCount * 3, reinterpret_cast<const float*>(params->vertexPositions),
                               size_t(params->vertexCount), size_t(params->vertexStride),
                               size_t(params->targetTriangleCount) * 3, std::numeric_limits<float>::max(), &quadricError)
        / 3;
#else
    // fallback to just dropping triangles
    size_t simplifiedTriangleCount = params->targetTriangleCount;
    std::copy_n(params->triangleVertices, params->targetTriangleCount, params->decimatedTriangleVertices);
#endif
    invoked = true;
    *result = nvclusterlod_DecimateTrianglesCallbackResult{.decimatedTriangleCount = static_cast<uint32_t>(simplifiedTriangleCount),
                                                           .additionalVertexCount = 0,
                                                           .quadricError          = quadricError};
    return NVCLUSTER_TRUE;
  };

  // The above lambda includes a capture, so it can't be passed through the C
  // API directly. This wrapper demonstrates passing it as the user data
  // instead. The '+' operator forces the wrapper to be a regular function
  // pointer, but is not strictly needed.
  using DecimateFn = decltype(decimateWithCapture);
  auto callback    = +[](void* userData, const nvclusterlod_DecimateTrianglesCallbackParams* params,
                      nvclusterlod_DecimateTrianglesCallbackResult* result) -> nvcluster_Bool {
    auto* fn = reinterpret_cast<DecimateFn*>(userData);
    return (*fn)(params, result);
  };

  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = static_cast<uint32_t>(icosphere.positions.size()),
      .vertexStride     = sizeof(vec3f),
      .clusterConfig =
          {
              .minClusterSize    = 16,
              .maxClusterSize    = 32,
              .costUnderfill     = 0.9f,
              .costOverlap       = 0.5f,
              .preSplitThreshold = 1u << 17,
          },
      .groupConfig =
          {
              .minClusterSize    = 8,
              .maxClusterSize    = 16,
              .costUnderfill     = 0.5f,
              .costOverlap       = 0.0f,
              .preSplitThreshold = 0,
          },
      .decimationFactor          = 0.5f,
      .userData                  = &decimateWithCapture,
      .decimateTrianglesCallback = callback,
  };

  nvclusterlod::LocalizedLodMesh mesh;
  nvclusterlod_Result            result = nvclusterlod::generateLocalizedLodMesh(lodContext.context, meshInput, mesh);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  EXPECT_TRUE(invoked);
}

TEST(Mesh, DecimateTrianglesCallbackWithAdditionalVertices)
{
  ScopedContext    context;
  ScopedLodContext lodContext(context);
  GeometryMesh     icosphere{makeIcosphere(2)};

  // Pre-allocate enough space for all additional vertices
  std::atomic<uint32_t> vertexCount         = uint32_t(icosphere.positions.size());
  uint32_t              originalVertexCount = vertexCount;
  icosphere.positions.resize(vertexCount * 3);

  // Provide a decimation callback that uses meshopt_simplifySloppy
  auto decimateWithCapture = [&icosphere, &vertexCount](const nvclusterlod_DecimateTrianglesCallbackParams* params,
                                                        nvclusterlod_DecimateTrianglesCallbackResult* result) -> nvcluster_Bool {
    float quadricError = 0.0f;
#if TESTS_HAVE_MESHOPTIMIZER
    size_t simplifiedTriangleCount =
        meshopt_simplifySloppy(reinterpret_cast<unsigned int*>(params->decimatedTriangleVertices),
                               reinterpret_cast<const unsigned int*>(params->triangleVertices),
                               params->triangleCount * 3, reinterpret_cast<const float*>(params->vertexPositions),
                               size_t(params->vertexCount), size_t(params->vertexStride),
                               size_t(params->targetTriangleCount) * 3, std::numeric_limits<float>::max(), &quadricError)
        / 3;
#else
    // fallback to just dropping triangles
    size_t simplifiedTriangleCount = params->targetTriangleCount;
    std::copy_n(params->triangleVertices, params->targetTriangleCount, params->decimatedTriangleVertices);
#endif

    // To exercise the API with adding new vertices during decimation, duplicate the inerior vertices (non-locked)
    auto triangles = std::span(reinterpret_cast<vec3u*>(params->decimatedTriangleVertices), simplifiedTriangleCount);
    std::map<uint32_t, uint32_t> interiorVertexDiplicates;
    for(const vec3u& triangle : triangles)
    {
      for(uint32_t vertexIndex : triangle)
      {
        if(!params->vertexLockFlags[vertexIndex])
        {
          interiorVertexDiplicates.try_emplace(vertexIndex, uint32_t(interiorVertexDiplicates.size()));
        }
      }
    }
    uint32_t additionalVertexCount  = uint32_t(interiorVertexDiplicates.size());
    uint32_t additionalVertexOffset = vertexCount.fetch_add(additionalVertexCount);
    if(additionalVertexOffset + additionalVertexCount > icosphere.positions.size())
    {
      // likely calls std::terminate()
      throw std::runtime_error("Not enough space for additional vertices");
    }
    for(auto [vertexIndex, duplicateIndex] : interiorVertexDiplicates)
    {
      icosphere.positions[additionalVertexOffset + duplicateIndex] = icosphere.positions[vertexIndex];
    }
    for(vec3u& triangle : triangles)
    {
      for(uint32_t& vertexIndex : triangle)
      {
        if(!params->vertexLockFlags[vertexIndex])
        {
          vertexIndex = additionalVertexOffset + interiorVertexDiplicates[vertexIndex];
        }
      }
    }
    *result = nvclusterlod_DecimateTrianglesCallbackResult{.decimatedTriangleCount = static_cast<uint32_t>(simplifiedTriangleCount),
                                                           .additionalVertexCount = additionalVertexCount,
                                                           .quadricError          = quadricError};
    return NVCLUSTER_TRUE;
  };

  // The above lambda includes a capture, so it can't be passed through the C
  // API directly. This wrapper demonstrates passing it as the user data
  // instead. The '+' operator forces the wrapper to be a regular function
  // pointer, but is not strictly needed.
  using DecimateFn = decltype(decimateWithCapture);
  auto callback    = +[](void* userData, const nvclusterlod_DecimateTrianglesCallbackParams* params,
                      nvclusterlod_DecimateTrianglesCallbackResult* result) -> nvcluster_Bool {
    auto* fn = reinterpret_cast<DecimateFn*>(userData);
    return (*fn)(params, result);
  };

  nvclusterlod_MeshInput meshInput{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(icosphere.triangles.data()),
      .triangleCount    = static_cast<uint32_t>(icosphere.triangles.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(icosphere.positions.data()),
      .vertexCount      = vertexCount,
      .vertexStride     = sizeof(vec3f),
      .clusterConfig =
          {
              .minClusterSize    = 16,
              .maxClusterSize    = 32,
              .costUnderfill     = 0.9f,
              .costOverlap       = 0.5f,
              .preSplitThreshold = 1u << 17,
          },
      .groupConfig =
          {
              .minClusterSize    = 8,
              .maxClusterSize    = 16,
              .costUnderfill     = 0.5f,
              .costOverlap       = 0.0f,
              .preSplitThreshold = 0,
          },
      .decimationFactor          = 0.5f,
      .userData                  = &decimateWithCapture,
      .decimateTrianglesCallback = callback,
  };

  nvclusterlod::LocalizedLodMesh mesh;
  nvclusterlod_Result            result = nvclusterlod::generateLocalizedLodMesh(lodContext.context, meshInput, mesh);
  ASSERT_EQ(result, nvclusterlod_Result::NVCLUSTERLOD_SUCCESS);
  EXPECT_GT(vertexCount, originalVertexCount) << "No new vertices were added";
  EXPECT_TRUE(std::ranges::any_of(mesh.vertexGlobalIndices, [originalVertexCount](uint32_t vertexIndex) {
    return vertexIndex >= originalVertexCount;
  })) << "No new vertices were referenced";
}

extern "C" int runCTest(void);

TEST(CAPI, Contexts)
{
  EXPECT_EQ(runCTest(), 1);
}
