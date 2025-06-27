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
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../nv_cluster_builder/src/clusters_cpp.hpp"  // reuse internal utils
#include <nvcluster/nvcluster.h>

// clang-format off
inline nvcluster::vec3f fromAPI(const nvcluster_Vec3f& v) { return {v.x, v.y, v.z}; }
inline nvcluster::vec3u fromAPI(const nvclusterlod_Vec3u& v) { return {v.x, v.y, v.z}; }
inline nvcluster_Vec3f toAPI(const nvcluster::vec3f& v) { return {v[0], v[1], v[2]}; }
inline nvclusterlod_Vec3u toAPI(const nvcluster::vec3u& v) { return {v[0], v[1], v[2]}; }
// clang-format on
