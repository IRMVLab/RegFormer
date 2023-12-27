from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_conv_select_k",
    ext_modules=[
        CUDAExtension(
            "fused_conv_select_k_cuda",
            ["fused_conv_g.cpp", "fused_conv_go.cu"],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']})
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)