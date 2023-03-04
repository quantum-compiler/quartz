/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      printf("Error: %s in line %d\n", custatevecGetErrorString(err),          \
             __LINE__);                                                        \
      return err;                                                              \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__);     \
      return err;                                                              \
    }                                                                          \
  };
