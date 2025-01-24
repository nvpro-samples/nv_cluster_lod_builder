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

#include <vector>

#include <nvclusterlod/nvclusterlod_hierarchy.h>


namespace nvclusterlod {

// Shortcut and storage for hierarchy output
struct LodHierarchy
{
  std::vector<Node>   nodes;
  std::vector<Sphere> groupCumulativeBoundingSpheres;
  std::vector<float>  groupCumulativeQuadricError;
};


inline nvclusterlod::Result generateLodHierarchy(nvclusterlod::Context context, const HierarchyInput& input, LodHierarchy& hierarchy)
{
  nvclusterlod::HierarchyGetRequirementsInfo reqInfo;
  reqInfo.input = &input;

  // Get conservative output sizes
  nvclusterlod::HierarchyRequirements sizes = nvclusterlodHierarchyGetRequirements(context, &reqInfo);

  // Allocate storage
  hierarchy.nodes.resize(sizes.maxNodeCount);
  hierarchy.groupCumulativeBoundingSpheres.resize(input.groupCount);
  hierarchy.groupCumulativeQuadricError.resize(input.groupCount);

  // Pack output pointers
  HierarchyOutput output;

  output.groupCumulativeBoundingSpheres = hierarchy.groupCumulativeBoundingSpheres.data();
  output.groupCumulativeQuadricError    = hierarchy.groupCumulativeQuadricError.data();
  output.nodeCount                      = sizes.maxNodeCount;
  output.nodes                          = hierarchy.nodes.data();


  HierarchyCreateInfo createInfo{&input};

  // Make LODs
  nvclusterlod::Result result = nvclusterlodCreateHierarchy(context, &createInfo, &output);
  if(result != nvclusterlod::Result::SUCCESS)
  {
    hierarchy = {};
    return result;
  }
  // Truncate to output size written
  hierarchy.nodes.resize(output.nodeCount);
  return nvclusterlod::Result::SUCCESS;
}

}  // namespace nvclusterlod
