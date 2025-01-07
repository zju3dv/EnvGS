#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

#include "common.h"
#include "optix_wrapper.h"


void createPipeline(const OptixDeviceContext optix_context, const std::string& pkg_dir, const std::string& pkg_name,
                    const char* raygen_name, const char* hitgroup_name,
                    OptixModule* module, OptixPipeline* pipeline, OptixShaderBindingTable& sbt)
{
    // -------------------------------------------------------
    // Get compiled PTX path
    std::string ptx_filename = pkg_dir + pkg_name;
    std::ifstream ptx_in(ptx_filename);
    if (!ptx_in) {
        std::cerr << "ERROR: readPTX() Failed to open file " << ptx_filename
                << std::endl;
        return;
    }
    // Load PTX from file
    std::string ptx = std::string((std::istreambuf_iterator<char>(ptx_in)),
                                   std::istreambuf_iterator<char>());

    OptixPipelineCompileOptions pipeline_compile_options = {};
    // -------------------------------------------------------
    // Define module and pipeline compile options
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    // Create module from PTX
    OPTIX_CHECK(optixModuleCreate(optix_context, &module_compile_options,
                                  &pipeline_compile_options, ptx.c_str(),
                                  ptx.size(), nullptr, nullptr, module));

    // -------------------------------------------------------
    // Create program groups
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    // Create raygen program group
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = *module;
    raygen_prog_group_desc.raygen.entryFunctionName = raygen_name;
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &raygen_prog_group_desc, 1,  // num program groups
                                        &program_group_options, nullptr, nullptr,
                                        &raygen_prog_group));
    // Create miss program group
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = nullptr;
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &miss_prog_group_desc, 1,  // num program groups
                                        &program_group_options, nullptr, nullptr,
                                        &miss_prog_group));
    // Create hitgroup program group
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleAH = *module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = hitgroup_name;
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &hitgroup_prog_group_desc, 1,  // num program groups
                                        &program_group_options, nullptr, nullptr,
                                        &hitgroup_prog_group));

    // -------------------------------------------------------
    // Link pipeline
    const uint32_t max_trace_depth = 31;
    OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};
    // Create pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    // OptiX 7.7.0 and later has removed the debugLevel field from OptixPipelineLinkOptions
    // pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
    OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline_compile_options,
                                    &pipeline_link_options, program_groups,
                                    sizeof(program_groups) / sizeof(program_groups[0]),
                                    nullptr, nullptr, pipeline));
    // Set stack size
    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, *pipeline));
    }
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0,  // maxCCDepth
                                           0,  // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(*pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          1));  // maxTraversableDepth

    // -------------------------------------------------------
    // Set up shader binding table
    // Raygen record
    void *raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record),
                          &rg_sbt,
                          raygen_record_size,
                          cudaMemcpyHostToDevice));
    // Miss record
    void *miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record),
                          &ms_sbt,
                          miss_record_size,
                          cudaMemcpyHostToDevice));
    // Hitgroup record
    void *hitgroup_record;
    const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record),
                          &hg_sbt,
                          hitgroup_record_size,
                          cudaMemcpyHostToDevice));
    // Set SBT
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;
}


OptiXStateWrapper::OptiXStateWrapper(const std::string& pkg_dir)
{
    // Create and allocate OptiX state
    optixState = new OptiXState();
    memset(optixState, 0, sizeof(OptiXState));

    // Initialize OptiX context
    optixState->context = nullptr;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    // 0 disable Setting the callback level will disable all messages. 
    // 1 fatal A non-recoverable error. 
    // 2 error A recoverable error, e.g., when passing invalid call parameters. 
    // 3 warning Hints that OptiX might not behave exactly as requested by the user or may perform slower than expected. 
    // 4 print Status or progress messages.
    options.logCallbackLevel = 3;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    // Associate a CUDA context (and therefore a specific GPU) with this device context
    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixState->context));

    // Create forward OptiX pipeline
    optixState->module_surfel_tracing_forward = nullptr;
    optixState->pipelie_surfel_tracing_forward = nullptr;
    optixState->sbt_surfel_tracing_forward = {};
    createPipeline(optixState->context, pkg_dir, "/forward.ptx", "__raygen__ot", "__anyhit__ot",
                   &optixState->module_surfel_tracing_forward, &optixState->pipelie_surfel_tracing_forward,
                   optixState->sbt_surfel_tracing_forward);

    // Create backward OptiX pipeline
    optixState->module_surfel_tracing_backward = nullptr;
    optixState->pipelie_surfel_tracing_backward = nullptr;
    optixState->sbt_surfel_tracing_backward = {};
    createPipeline(optixState->context, pkg_dir, "/backward.ptx", "__raygen__ot", "__anyhit__ot",
                   &optixState->module_surfel_tracing_backward, &optixState->pipelie_surfel_tracing_backward,
                   optixState->sbt_surfel_tracing_backward);
}


OptiXStateWrapper::~OptiXStateWrapper()
{
    OPTIX_CHECK(optixPipelineDestroy(optixState->pipelie_surfel_tracing_forward));
    OPTIX_CHECK(optixPipelineDestroy(optixState->pipelie_surfel_tracing_backward));
    OPTIX_CHECK(optixModuleDestroy(optixState->module_surfel_tracing_forward));
    OPTIX_CHECK(optixModuleDestroy(optixState->module_surfel_tracing_backward));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_forward.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_forward.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_forward.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_backward.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_backward.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->sbt_surfel_tracing_backward.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixState->d_gas_output_buffer)));
    OPTIX_CHECK(optixDeviceContextDestroy(optixState->context));
    delete optixState;
}
