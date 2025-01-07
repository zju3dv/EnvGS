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

# Utility function to get PTX compilation & copying working
function(add_ptx_targets project library names)
  # Target to copy PTX files
  set(LIBRARY_NAMES ${names})
  list(TRANSFORM LIBRARY_NAMES PREPEND ${project}_)
  add_custom_target(
    ${project}_copy_ptx ALL
    COMMENT "Copy PTX Files for ${project}"
    DEPENDS ${LIBRARY_NAMES})

  # Target to create PTX directory
  add_custom_command(TARGET ${project}_copy_ptx PRE_BUILD
    COMMENT "Create directory for PTX files for ${project}"
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${project}>/ptx)

  # Create PTX objects
  foreach(cur_name ${names})
    # add_library(${project}_${cur_name} OBJECT src/${cur_name}.cu)
    add_library(${project}_${cur_name} OBJECT ${cur_name}.cu)
    set_target_properties(${project}_${cur_name} PROPERTIES CUDA_PTX_COMPILATION ON)
    add_dependencies(${project} ${project}_${cur_name})

    # Add current PTX to copy target
    add_custom_command(TARGET ${project}_copy_ptx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${project}_${cur_name}> $<TARGET_FILE_DIR:${project}>/ptx
      DEPENDS ${project}_${cur_name})
  endforeach()
endfunction()
