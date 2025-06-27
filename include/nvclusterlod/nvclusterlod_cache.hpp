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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <cassert>
#include <cstring>

#include "nvclusterlod_common.h"
#include "nvclusterlod_hierarchy_storage.hpp"
#include "nvclusterlod_mesh_storage.hpp"

// This file provides basic helpers to de-/serialize the
// key containers from the storage classes into a flat
// uncompressed binary cache.

namespace nvclusterlod {

namespace detail {
static constexpr uint64_t ALIGNMENT  = 16ULL;
static constexpr uint64_t ALIGN_MASK = ALIGNMENT - 1;
static_assert(ALIGNMENT >= sizeof(uint64_t));
template <typename T>
inline uint64_t getCachedSize(const std::span<T>& view)
{
  // use one extra ALIGNMENT to store count
  return ((view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK) + ALIGNMENT;
}

template <typename T>
inline void storeAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, const std::span<const T>& view)
{
  assert(static_cast<uint64_t>(dataAddress) % ALIGNMENT == 0);

  if(isValid && dataAddress + getCachedSize(view) <= dataEnd)
  {
    union
    {
      uint64_t count;
      uint8_t  countData[ALIGNMENT];
    };
    memset(countData, 0, sizeof(countData));

    count = view.size();

    // store count first
    memcpy(reinterpret_cast<void*>(dataAddress), countData, ALIGNMENT);
    dataAddress += ALIGNMENT;
    
    if(view.size())
    {
      // then data
      memcpy(reinterpret_cast<void*>(dataAddress), view.data(), view.size_bytes());
      dataAddress += (view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK;
    }
  }
  else
  {
    isValid = false;
  }
}

template <typename T>
inline void loadAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, std::span<const T>& view)
{
  union
  {
    const T* basePointer;
    uint64_t baseRaw;
  };
  baseRaw = dataAddress;

  assert(dataAddress % ALIGNMENT == 0);

  uint64_t count = *reinterpret_cast<const uint64_t*>(basePointer);
  baseRaw += ALIGNMENT;

  if(isValid && count && (baseRaw + (sizeof(T) * count) <= dataEnd))
  {
    // each array is 16 byte aligned
    view = std::span<const T>(basePointer, count);
  }
  else
  {
    view = {};
    // count of zero is valid, otherwise bail
    isValid = isValid && count == 0;
  }

  baseRaw += sizeof(T) * count;
  baseRaw = (baseRaw + ALIGN_MASK) & ~(ALIGN_MASK);

  dataAddress = baseRaw;
}
}  // namespace detail

struct LodMeshView
{
  std::span<const nvclusterlod_Vec3u>  triangleVertices;
  std::span<const nvcluster_Range>     clusterTriangleRanges;
  std::span<const uint32_t>            clusterGeneratingGroups;
  std::span<const nvclusterlod_Sphere> clusterBoundingSpheres;
  std::span<const float>               groupQuadricErrors;
  std::span<const nvcluster_Range>     groupClusterRanges;
  std::span<const nvcluster_Range>     lodLevelGroupRanges;
};

inline void toView(const LodMesh& storage, LodMeshView& view)
{
  view.triangleVertices        = storage.triangleVertices;
  view.clusterTriangleRanges   = storage.clusterTriangleRanges;
  view.clusterGeneratingGroups = storage.clusterGeneratingGroups;
  view.clusterBoundingSpheres  = storage.clusterBoundingSpheres;
  view.groupQuadricErrors      = storage.groupQuadricErrors;
  view.groupClusterRanges      = storage.groupClusterRanges;
  view.lodLevelGroupRanges     = storage.lodLevelGroupRanges;
}

inline void toStorage(const LodMeshView& view, LodMesh& storage)
{
  storage.triangleVertices.resize(view.triangleVertices.size());
  storage.clusterTriangleRanges.resize(view.clusterTriangleRanges.size());
  storage.clusterGeneratingGroups.resize(view.clusterGeneratingGroups.size());
  storage.clusterBoundingSpheres.resize(view.clusterBoundingSpheres.size());
  storage.groupQuadricErrors.resize(view.groupQuadricErrors.size());
  storage.groupClusterRanges.resize(view.groupClusterRanges.size());
  storage.lodLevelGroupRanges.resize(view.lodLevelGroupRanges.size());

  memcpy(storage.triangleVertices.data(), view.triangleVertices.data(), view.triangleVertices.size_bytes());
  memcpy(storage.clusterTriangleRanges.data(), view.clusterTriangleRanges.data(), view.clusterTriangleRanges.size_bytes());
  memcpy(storage.clusterGeneratingGroups.data(), view.clusterGeneratingGroups.data(), view.clusterGeneratingGroups.size_bytes());
  memcpy(storage.clusterBoundingSpheres.data(), view.clusterBoundingSpheres.data(), view.clusterBoundingSpheres.size_bytes());
  memcpy(storage.groupQuadricErrors.data(), view.groupQuadricErrors.data(), view.groupQuadricErrors.size_bytes());
  memcpy(storage.groupClusterRanges.data(), view.groupClusterRanges.data(), view.groupClusterRanges.size_bytes());
  memcpy(storage.lodLevelGroupRanges.data(), view.lodLevelGroupRanges.data(), view.lodLevelGroupRanges.size_bytes());
}

inline uint64_t getCachedSize(const LodMeshView& meshView)
{
  uint64_t cachedSize = 0;

  cachedSize += detail::getCachedSize(meshView.triangleVertices);
  cachedSize += detail::getCachedSize(meshView.clusterTriangleRanges);
  cachedSize += detail::getCachedSize(meshView.clusterGeneratingGroups);
  cachedSize += detail::getCachedSize(meshView.clusterBoundingSpheres);
  cachedSize += detail::getCachedSize(meshView.groupQuadricErrors);
  cachedSize += detail::getCachedSize(meshView.groupClusterRanges);
  cachedSize += detail::getCachedSize(meshView.lodLevelGroupRanges);

  return cachedSize;
}

inline bool storeCached(const LodMeshView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.triangleVertices);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterTriangleRanges);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterGeneratingGroups);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterBoundingSpheres);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupQuadricErrors);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupClusterRanges);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.lodLevelGroupRanges);

  return isValid;
}

inline bool loadCached(LodMeshView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.triangleVertices);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterTriangleRanges);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterGeneratingGroups);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterBoundingSpheres);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupQuadricErrors);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupClusterRanges);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.lodLevelGroupRanges);

  return isValid;
}

struct LodHierarchyView
{
  std::span<const nvclusterlod_HierarchyNode> nodes;
  std::span<const nvclusterlod_Sphere>        groupCumulativeBoundingSpheres;
  std::span<const float>                      groupCumulativeQuadricError;
};

inline void toView(const LodHierarchy& storage, LodHierarchyView& view)
{
  view.nodes                          = storage.nodes;
  view.groupCumulativeBoundingSpheres = storage.groupCumulativeBoundingSpheres;
  view.groupCumulativeQuadricError    = storage.groupCumulativeQuadricError;
}

inline void toStorage(const LodHierarchyView& view, LodHierarchy& storage)
{
  storage.nodes.resize(view.nodes.size());
  storage.groupCumulativeBoundingSpheres.resize(view.groupCumulativeBoundingSpheres.size());
  storage.groupCumulativeQuadricError.resize(view.groupCumulativeQuadricError.size());

  memcpy(storage.nodes.data(), view.nodes.data(), view.nodes.size_bytes());
  memcpy(storage.groupCumulativeBoundingSpheres.data(), view.groupCumulativeBoundingSpheres.data(),
         view.groupCumulativeBoundingSpheres.size_bytes());
  memcpy(storage.groupCumulativeQuadricError.data(), view.groupCumulativeQuadricError.data(),
         view.groupCumulativeQuadricError.size_bytes());
}

inline LodHierarchyView getView(const LodHierarchy& hierarchy)
{
  LodHierarchyView view;
  view.nodes                          = hierarchy.nodes;
  view.groupCumulativeBoundingSpheres = hierarchy.groupCumulativeBoundingSpheres;
  view.groupCumulativeQuadricError    = hierarchy.groupCumulativeQuadricError;
  return view;
}

inline uint64_t getCachedSize(const LodHierarchyView& hierarchyView)
{
  uint64_t cachedSize = 0;

  cachedSize += detail::getCachedSize(hierarchyView.nodes);
  cachedSize += detail::getCachedSize(hierarchyView.groupCumulativeBoundingSpheres);
  cachedSize += detail::getCachedSize(hierarchyView.groupCumulativeQuadricError);

  return cachedSize;
}

inline bool storeCached(const LodHierarchyView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.nodes);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupCumulativeBoundingSpheres);
  detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupCumulativeQuadricError);

  return isValid;
}

inline bool loadCached(LodHierarchyView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.nodes);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupCumulativeBoundingSpheres);
  detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupCumulativeQuadricError);

  return isValid;
}

struct LodGeometryInfo
{
  // details of the original mesh are embedded for compatibility
  uint64_t         inputTriangleCount       = 0;
  uint64_t         inputVertexCount         = 0;
  uint64_t         inputTriangleIndicesHash = 0;
  uint64_t         inputVerticesHash        = 0;
  nvcluster_Config clusterConfig;
  nvcluster_Config groupConfig;
  float            decimationFactor = 0;
};

struct LodGeometryView
{
  LodGeometryInfo info;

  // this is also the storage order
  LodMeshView      lodMesh;
  LodHierarchyView lodHierarchy;
};

inline uint64_t getCachedSize(const LodGeometryView& view)
{
  uint64_t cachedSize = 0;

  cachedSize += (sizeof(LodGeometryInfo) + detail::ALIGN_MASK) & ~detail::ALIGN_MASK;
  cachedSize += getCachedSize(view.lodMesh);
  cachedSize += getCachedSize(view.lodHierarchy);

  return cachedSize;
}

inline bool storeCached(const LodGeometryView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = dataAddress % detail::ALIGNMENT == 0 && dataAddress + sizeof(LodGeometryInfo) <= dataEnd;

  if(isValid)
  {
    memcpy(reinterpret_cast<void*>(dataAddress), &view.info, sizeof(LodGeometryInfo));
    dataAddress += (sizeof(LodGeometryInfo) + detail::ALIGN_MASK) & ~detail::ALIGN_MASK;
  }

  isValid = isValid && storeCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += getCachedSize(view.lodMesh);
  isValid = isValid && storeCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += getCachedSize(view.lodHierarchy);

  return isValid;
}

inline bool loadCached(LodGeometryView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  if(dataAddress % detail::ALIGNMENT == 0 && dataAddress + sizeof(LodGeometryInfo) <= dataEnd)
  {
    memcpy(&view.info, data, sizeof(LodGeometryInfo));
    dataAddress += (sizeof(LodGeometryInfo) + detail::ALIGN_MASK) & ~detail::ALIGN_MASK;
  }
  else
  {
    view = {};
    return false;
  }

  isValid = isValid && loadCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<const void*>(dataAddress));
  dataAddress += getCachedSize(view.lodMesh);
  isValid = isValid && loadCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<const void*>(dataAddress));
  dataAddress += getCachedSize(view.lodHierarchy);
  return isValid;
}

class CacheHeader
{
public:
  CacheHeader()
  {
    std::fill(std::begin(data), std::end(data), 0);
    header = {};
  }

private:
  struct Header
  {
    uint64_t magic          = 0x00646f6c6c63766eULL;  // nvcllod
    uint32_t lodVersion     = NVCLUSTERLOD_VERSION;
    uint32_t clusterVersion = NVCLUSTER_VERSION;
  };

  union
  {
    Header  header;
    uint8_t data[(sizeof(Header) + detail::ALIGNMENT - 1) & ~(detail::ALIGNMENT - 1)];
  };
};

class CacheView
{
  // Optionally if you want to have a simple cache file for this
  // data, we provide a canonical layout, and this simple class
  // to open it.
  //
  // The cache data must be stored in three sections:
  //
#if 0
  struct CacheFile
  {
    // first: library version specific header
    CacheHeader header;
    // second: for each geometry serialized data of the `LodGeometryView`
    uint8_t geometryViewData[];
    // third: offset table
    // offsets where each `LodGeometry` data is stored.
    // ordered with ascending offsets
    // `geometryDataSize = geometryOffsets[geometryIndex + 1] - geometryOffsets[geometryIndex];`
    uint64_t geometryOffsets[geometryCount + 1];
    uint64_t geometryCount;
  };
#endif

public:
  bool isValid() const
  {
    return m_dataSize != 0;
  }

  bool init(uint64_t dataSize, const void* data)
  {
    m_dataSize  = dataSize;
    m_dataBytes = reinterpret_cast<const uint8_t*>(data);

    if(dataSize <= sizeof(CacheHeader) + sizeof(uint64_t))
    {
      m_dataSize = 0;
      return false;
    }

    CacheHeader defaultHeader;

    if(memcmp(data, &defaultHeader, sizeof(CacheHeader)) != 0)
    {
      m_dataSize = 0;
      return false;
    }

    m_geometryCount = *getPointer<uint64_t>(m_dataSize - sizeof(uint64_t));

    if(dataSize <= (sizeof(CacheHeader) + sizeof(uint64_t) * (m_geometryCount + 2)))
    {
      m_dataSize = 0;
      return false;
    }

    m_tableStart = m_dataSize - sizeof(uint64_t) * (m_geometryCount + 2);

    return true;
  }

  void deinit()
  {
    *(this) = {};
  }

  uint64_t getGeometryCount() const
  {
    return m_geometryCount;
  }

  bool getLodGeometryView(LodGeometryView& view, uint64_t geometryIndex) const
  {
    if(geometryIndex >= m_geometryCount)
    {
      assert(0);
      return false;
    }

    const uint64_t* geometryOffsets = getPointer<uint64_t>(m_tableStart, m_geometryCount + 1);
    uint64_t        base            = geometryOffsets[geometryIndex];

    if(base + sizeof(LodGeometryInfo) > m_tableStart)
    {
      // this must not happen on a valid file
      assert(0);
      return false;
    }

    uint64_t geometryTotalSize = geometryOffsets[geometryIndex + 1] - base;

    const uint8_t* geoData = getPointer<uint8_t>(base, geometryTotalSize);

    return loadCached(view, geometryTotalSize, geoData);
  }

private:
  template <class T>
  const T* getPointer(uint64_t offset, [[maybe_unused]] uint64_t count = 1) const
  {
    assert(offset + sizeof(T) * count <= m_dataSize);
    return reinterpret_cast<const T*>(m_dataBytes + offset);
  }

  uint64_t       m_dataSize      = 0;
  uint64_t       m_tableStart    = 0;
  const uint8_t* m_dataBytes     = nullptr;
  uint64_t       m_geometryCount = 0;
};
}  // namespace nvclusterlod
