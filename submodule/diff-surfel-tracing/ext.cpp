#include <torch/extension.h>

#include "trace_surfels.h"
#include "optix_tracer/optix_wrapper.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper").def(pybind11::init<const std::string &>());
  m.def("build_acceleration_structure", &BuildAccelerationStructure);
  m.def("trace_surfels", &TraceSurfelsCUDA);
  m.def("trace_surfels_backward", &TraceSurfelsBackwardCUDA);
}
