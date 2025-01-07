# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Define the project
project(optix_tracer LANGUAGES CXX CUDA)

# Include directories
include_directories(./)

# Add the library
add_library(optix_tracer_forward OBJECT forward.cu)
# Set the property for PTX compilation
set_property(TARGET optix_tracer_forward PROPERTY CUDA_PTX_COMPILATION ON)
# Get the path to the forward.cu file
get_filename_component(FORWARD_CU_PATH "${CMAKE_CURRENT_SOURCE_DIR}/forward.cu" ABSOLUTE)
# Get the path to the forward.ptx file
get_filename_component(FORWARD_CU_DIR "${FORWARD_CU_PATH}" DIRECTORY)
# file(RELATIVE_PATH REL_FORWARD_PTX_PATH "${CMAKE_BINARY_DIR}" "${FORWARD_CU_DIR}")
set(FORWARD_PTX_PATH "${CMAKE_BINARY_DIR}/optix_tracer/CMakeFiles/optix_tracer_forward.dir/forward.ptx")
# Add a custom command to copy the forward.ptx file to the ptx directory
add_custom_target(copy_forward_ptx_to_ptx_dir
    ALL  # This command is always run
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/ptx
    COMMAND ${CMAKE_COMMAND} -E copy
            ${FORWARD_PTX_PATH}
            ${CMAKE_BINARY_DIR}/ptx/forward.ptx
    COMMENT "Copying forward.ptx to ptx directory"
)
# Copy only after the forward.ptx file is generated
add_dependencies(copy_forward_ptx_to_ptx_dir optix_tracer_forward)

# Add the library
add_library(optix_tracer_backward OBJECT backward.cu)
# Set the property for PTX compilation
set_property(TARGET optix_tracer_backward PROPERTY CUDA_PTX_COMPILATION ON)
# Get the path to the backward.cu file
get_filename_component(BACKWARD_CU_PATH "${CMAKE_CURRENT_SOURCE_DIR}/backward.cu" ABSOLUTE)
# Get the path to the backward.ptx file
get_filename_component(BACKWARD_CU_DIR "${BACKWARD_CU_PATH}" DIRECTORY)
# file(RELATIVE_PATH REL_BACKWARD_PTX_PATH "${CMAKE_BINARY_DIR}" "${BACKWARD_CU_DIR}")
set(BACKWARD_PTX_PATH "${CMAKE_BINARY_DIR}/optix_tracer/CMakeFiles/optix_tracer_backward.dir/backward.ptx")
# Add a custom command to copy the backward.ptx file to the ptx directory
add_custom_target(copy_backward_ptx_to_ptx_dir
    ALL  # This command is always run
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/ptx
    COMMAND ${CMAKE_COMMAND} -E copy
            ${BACKWARD_PTX_PATH}
            ${CMAKE_BINARY_DIR}/ptx/backward.ptx
    COMMENT "Copying backward.ptx to ptx directory"
)
# Copy only after the backward.ptx file is generated
add_dependencies(copy_backward_ptx_to_ptx_dir optix_tracer_backward)
