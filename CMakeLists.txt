# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(nv_cluster_lod_builder VERSION 2.0)

option(NVCLUSTERLOD_MULTITHREADED "Use parallel algorithms to generate clusters." ON)
set(NVCLUSTER_MULTITHREADED ${NVCLUSTERLOD_MULTITHREADED} CACHE BOOL "Build with multithreaded cluster generation support" FORCE)

if(NOT TARGET meshoptimizer)
  add_subdirectory(meshoptimizer)
endif()
if(NOT TARGET nv_cluster_builder)
  add_subdirectory(nv_cluster_builder)
endif()

set(SOURCES
  src/array_view.hpp
  src/array_view.natvis
  src/nvclusterlod_context.cpp
  src/nvclusterlod_context.hpp
  src/nvclusterlod_hierarchy.cpp
  src/nvclusterlod_mesh.cpp
  src/nvclusterlod_cpp.hpp
)
set(HEADERS
  include/nvclusterlod/nvclusterlod_cache.hpp
  include/nvclusterlod/nvclusterlod_common.h
  include/nvclusterlod/nvclusterlod_hierarchy.h
  include/nvclusterlod/nvclusterlod_hierarchy_storage.hpp
  include/nvclusterlod/nvclusterlod_mesh.h
  include/nvclusterlod/nvclusterlod_mesh_storage.hpp
)

source_group("public_include" FILES ${HEADERS})
source_group("source" FILES ${SOURCES})

# Optionally build as a shared library
include(CMakeDependentOption)
cmake_dependent_option(
  NVCLUSTERLOD_BUILDER_SHARED  # option variable
  "Build shared library"       # description
  OFF                          # default value if exposed; user can override
  "NOT BUILD_SHARED_LIBS"      # condition to expose option
  ON                           # value if not exposed; user can't override
)

if (NVCLUSTERLOD_BUILDER_SHARED)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
  add_library(nv_cluster_lod_builder SHARED ${SOURCES} ${HEADERS})
  target_compile_definitions(nv_cluster_lod_builder PUBLIC NVCLUSTERLOD_BUILDER_SHARED)
else()
  add_library(nv_cluster_lod_builder STATIC ${SOURCES} ${HEADERS})
endif ()
target_compile_features(nv_cluster_lod_builder PUBLIC cxx_std_20)
target_include_directories(nv_cluster_lod_builder PUBLIC include)
target_include_directories(nv_cluster_lod_builder PRIVATE src)
target_compile_definitions(nv_cluster_lod_builder PRIVATE NVCLUSTERLOD_BUILDER_COMPILING)

# All the warnings. Branch on COMPILE_LANGUAGE to avoid passing unknowns to nvcc
if(MSVC)
    target_compile_options(nv_cluster_lod_builder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
    )
    target_compile_definitions(nv_cluster_lod_builder PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_lod_builder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_lod_builder PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            # Do not use ABI breaking _GLIBCXX_DEBUG or _GLIBCXX_DEBUG_BACKTRACE
        )
    endif()
endif()

# Headers from nv_cluster_builder are used in the API
target_link_libraries(nv_cluster_lod_builder PUBLIC nv_cluster_builder)
target_link_libraries(nv_cluster_lod_builder PRIVATE meshoptimizer)
target_compile_definitions(nv_cluster_lod_builder PUBLIC NVCLUSTERLOD_MULTITHREADED=$<BOOL:${NVCLUSTERLOD_MULTITHREADED}>)

if(NOT MSVC)
  # Optional TBB for std::execution on linux
  find_library(TBB_LIBRARIES NAMES tbb HINTS ${TBB_DIR})
  if(TBB_LIBRARIES)
      message(STATUS "TBB: ${TBB_LIBRARIES}")
      target_link_libraries(nv_cluster_lod_builder PRIVATE ${TBB_LIBRARIES})
  else()
      message(STATUS "TBB not found for std::execution")
  endif()
endif()

if(BUILD_TESTING)
  option(BUILD_NV_CLUSTER_LOD_BUILDER_TESTING "Build nv_cluster_lod_builder tests" ON)
  if(BUILD_NV_CLUSTER_LOD_BUILDER_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()

install(TARGETS nv_cluster_lod_builder)
