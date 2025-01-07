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

# Try to locate the Optix installation using the OPTIX_HOME environment variable
find_path(OPTIX_HOME include/optix.h 
    PATHS
        ENV OPTIX_HOME
        ENV OPTIX_ROOT
        ${CMAKE_CURRENT_LIST_DIR}/../third_party/optix  # default OptiX path
    DOC "Path to Optix installation.")

if(${OPTIX_HOME} STREQUAL "OptiX7_HOME-NOTFOUND")
	if (${OptiX7_FIND_REQUIRED})
        message(FATAL_ERROR "OPTIX_HOME not defined")
	elseif(NOT ${OptiX7_FIND_QUIETLY})
        message(STATUS "OPTIX_HOME not defined")
	endif()
endif()

# Include
find_path(OptiX7_INCLUDE_DIR 
	NAMES optix.h
    PATHS "${OPTIX_HOME}/include"
	NO_DEFAULT_PATH
	)
find_path(OptiX7_INCLUDE_DIR
	NAMES optix.h
	)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX7 DEFAULT_MSG 
	OptiX7_INCLUDE_DIR)

set(OptiX7_INCLUDE_DIRS ${OptiX7_INCLUDE_DIR})
if(WIN32)
	set(OptiX7_DEFINITIONS NOMINMAX)
endif()
mark_as_advanced(OptiX7_INCLUDE_DIRS OptiX7_DEFINITIONS)

add_library(OptiX7 INTERFACE)
target_compile_definitions(OptiX7 INTERFACE ${OptiX7_DEFINITIONS})
target_include_directories(OptiX7 INTERFACE ${OptiX7_INCLUDE_DIRS})
if(NOT WIN32)
    target_link_libraries(OptiX7 INTERFACE dl)
endif()
