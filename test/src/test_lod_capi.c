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

#ifdef __cplusplus
#error This file verifies the API is C compatible
#endif

#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_mesh.h>
#include <stdio.h>

int runCTest(void)
{
  nvcluster_ContextCreateInfo clusterCreateInfo   = nvcluster_defaultContextCreateInfo();
  nvcluster_Context           clusterContext      = 0;
  nvcluster_Result            clusterCreateResult = nvclusterCreateContext(&clusterCreateInfo, &clusterContext);
  if(clusterCreateResult != NVCLUSTER_SUCCESS)
  {
    printf("Create Context Result: %s\n", nvclusterResultString(clusterCreateResult));
    return 0;
  }

  nvclusterlod_ContextCreateInfo lodCreateInfo = {
      .version        = NVCLUSTERLOD_VERSION,
      .parallelize    = NVCLUSTER_TRUE,
      .clusterContext = clusterContext,
  };
  nvclusterlod_Context lodContext      = 0;
  nvclusterlod_Result  lodCreateResult = nvclusterlodCreateContext(&lodCreateInfo, &lodContext);
  if(lodCreateResult != NVCLUSTERLOD_SUCCESS)
  {
    printf("Create Context Result: %s\n", nvclusterlodResultString(lodCreateResult));
    return 0;
  }

  nvclusterlod_Result lodDestroyResult = nvclusterlodDestroyContext(lodContext);
  if(lodDestroyResult != NVCLUSTERLOD_SUCCESS)
  {
    printf("Destroy Context Result: %s\n", nvclusterlodResultString(lodDestroyResult));
    return 0;
  }

  nvcluster_Result clusterDestroyResult = nvclusterDestroyContext(clusterContext);
  if(clusterDestroyResult != NVCLUSTER_SUCCESS)
  {
    printf("Destroy Context Result: %s\n", nvclusterResultString(clusterDestroyResult));
    return 0;
  }
  return 1;
}
