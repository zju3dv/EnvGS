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

#include "common.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include <optix.h>

void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata) {
  std::cout << "Optix Log[" << level << "][" << tag << "]: '" << message
            << "'\n";
}

void printActiveCudaDevices(void) {
  // Query number of available CUDA devices
  int num_cuda_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));

  // Print available CUDA devices' names
  std::cout << "Active CUDA Devices: \n";
  for (int i = 0; i < num_cuda_devices; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    std::cout << "\tDevice " << i << ": " << prop.name << "\n";
  }
  std::cout << "\n";
}

std::string getFileContent(std::string const &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file for reading: '" + path + "'");

  std::stringstream stream;
  stream << file.rdbuf();

  if (file.bad() || file.fail())
    throw std::runtime_error("Error reading file content from: '" + path + "'");

  return stream.str();
}

void cuda_free_event_callback(cudaStream_t stream, cudaError_t status,
                              void *userData) {
  cudaFree(userData);
}
