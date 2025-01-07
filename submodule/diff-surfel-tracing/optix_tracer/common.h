/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <optix.h>

#define OPTIX_CHECK(error)                                                     \
  {                                                                            \
    if (error != OPTIX_SUCCESS)                                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Optix Error: '"            \
                << optixGetErrorString(error) << "'\n";                        \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }

#define CUDA_DRIVER_CHECK(error)                                               \
  {                                                                            \
    if (error != CUDA_SUCCESS) {                                               \
      const char *error_str = nullptr;                                         \
      cuGetErrorString(error, &error_str);                                     \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << error_str << "'\n";                                         \
    }                                                                          \
  }

// This is used to construct the path to the PTX files
#ifdef _WIN32
#ifdef NDEBUG
#define BUILD_DIR "Release"
#else
#define BUILD_DIR "Debug"
#endif
#else
#define BUILD_DIR "./"
#endif

template <typename T> struct SbtRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

// template specialization for empty records
template <> struct SbtRecord<void> {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata);

void printActiveCudaDevices(void);

std::string getFileContent(std::string const &path);

void cuda_free_event_callback(cudaStream_t stream, cudaError_t status,
                              void *userData);
