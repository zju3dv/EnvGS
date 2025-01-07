import os
import site
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


# Determine the OptiX SDK path
OPTIX_HOME = os.environ.get('OPTIX_HOME')
if OPTIX_HOME is None or not os.path.exists(OPTIX_HOME):
    OPTIX_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/optix")
    if not os.path.exists(OPTIX_HOME):
        raise ValueError("Please set the OPTIX_HOME environment variable to the path to the OptiX SDK")
OPTIX_HOME = os.path.join(OPTIX_HOME, "include")
print(f"Using OptiX SDK at {OPTIX_HOME}")


# Custom build extension to build the OptiX tracing kernel
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Record the python package source directory
        pkg_source = os.path.dirname(os.path.abspath(__file__))

        # Run the original build_extensions
        super().build_extensions()

        # Use CMake to build the OptiX tracing kernel ptx files
        os.system(f'mkdir -p {pkg_source}/build && cd {pkg_source}/build && cmake .. && make')
        pkg_target = site.getsitepackages()[0] + '/diff_surfel_tracing'

        # Create the target directory if it does not exist
        if not os.path.exists(pkg_target):
            os.makedirs(pkg_target, exist_ok=True)

        # Copy the `.ptx` files to the python package
        os.system(f'cp {pkg_source}/build/ptx/*.ptx {pkg_target}')


# Setup for the python package
setup(
    name="diff_surfel_tracing",
    packages=['diff_surfel_tracing'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_tracing._C",
            sources=[
                "optix_tracer/common.cpp",
                "optix_tracer/optix_wrapper.cpp",
                "trace_surfels.cpp",
                "ext.cpp"
            ],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]},
            include_dirs=[OPTIX_HOME]
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)
