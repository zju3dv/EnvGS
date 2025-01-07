#pragma once

#include <string>
#include <iterator>
#include <optix.h>
#include <optix_stubs.h>

#include "common.h"
#include "params.h"


// Define roundUp function
template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}


// Typedef different SBT records
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


struct OptiXState
{
    // OptiX context
    OptixDeviceContext context = nullptr;
    OptixTraversableHandle gas_handle;
    void *d_gas_output_buffer;    

    // Forward module, SBT, pipeline
    OptixModule module_surfel_tracing_forward = nullptr;
    OptixShaderBindingTable sbt_surfel_tracing_forward = {};
    OptixPipeline pipelie_surfel_tracing_forward = nullptr;
    // Backward module, SBT, pipeline
    OptixModule module_surfel_tracing_backward = nullptr;
    OptixShaderBindingTable sbt_surfel_tracing_backward = {};
    OptixPipeline pipelie_surfel_tracing_backward = nullptr;
};


class OptiXStateWrapper
{
public:
    // OptiX state
    OptiXState* optixState;

    // Constructor and destructor
    OptiXStateWrapper(const std::string& pkg_dir);
    ~OptiXStateWrapper();
};
