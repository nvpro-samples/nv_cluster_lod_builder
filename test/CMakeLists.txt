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

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(nv_cluster_lod_builder_tests
  src/test_lod.cpp
  src/test_lod_capi.c
)
target_link_libraries(nv_cluster_lod_builder_tests nv_cluster_lod_builder gtest_main gmock_main)

# opportunistically 
if(TARGET meshoptimizer)
  target_link_libraries(nv_cluster_lod_builder_tests meshoptimizer)
  target_compile_definitions(nv_cluster_lod_builder_tests PRIVATE TESTS_HAVE_MESHOPTIMIZER=1)
else()
  target_compile_definitions(nv_cluster_lod_builder_tests PRIVATE TESTS_HAVE_MESHOPTIMIZER=0)
endif()

if(MSVC)
    target_compile_options(nv_cluster_lod_builder_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/WX>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4201> # nonstandard extension used: nameless struct/union
    )
    target_compile_definitions(nv_cluster_lod_builder_tests PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_lod_builder_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
      $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
      $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_lod_builder_tests PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            # Do not use ABI breaking _GLIBCXX_DEBUG or _GLIBCXX_DEBUG_BACKTRACE
        )
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(nv_cluster_lod_builder_tests)
