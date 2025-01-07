# FindTorch.cmake

# Try to determine the installation path of PyTorch using torch.utils.cmake prefix path
find_path(TORCH_CMAKE_PREFIX_PATH NAMES cmake/TorchConfig.cmake
  PATHS
    ENV TORCH_CMAKE_PREFIX_PATH
    DOC "Path to PyTorch cmake configuration."
)

# Use the python command to find it if the TORCH CMAKE PREFIX PATH is not found
if(NOT TORCH_CMAKE_PREFIX_PATH)
  execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

message(STATUS "Found PyTorch CMake Prefix Path: ${TORCH_CMAKE_PREFIX_PATH}")

# Make sure the TORCH CMAKE PREFIX PATH points to the correct path
if(NOT IS_DIRECTORY ${TORCH_CMAKE_PREFIX_PATH})
  message(FATAL_ERROR "Invalid PyTorch cmake prefix path: ${TORCH_CMAKE_PREFIX_PATH}")
endif()

# Include PyTorch configuration
include(${TORCH_CMAKE_PREFIX_PATH}/ATen/ATenConfig.cmake)
include(${TORCH_CMAKE_PREFIX_PATH}/Torch/TorchConfig.cmake)
# include(${TORCH_CMAKE_PREFIX_PATH}/Tensorpipe/TensorpipeConfig.cmake)
