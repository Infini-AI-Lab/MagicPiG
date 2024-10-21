from torch.utils.cpp_extension import CppExtension
import setuptools

ext_modules = [
    CppExtension(
        "gather_gemv_cpu",
        sources=[
            "gather_gemv.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16Convert.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16ConvertAvx2.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16ConvertAvx512.cc",
            "3rdparty/FBGEMM/src/RefImplementations.cc",
            "3rdparty/FBGEMM/src/Utils.cc",
        ],
        include_dirs=["3rdparty/FBGEMM/include"],
        extra_compile_args=["-mavx512f", "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0","-std=c++17"],
        extra_link_args=['-fopenmp'],
    ),
]

setuptools.setup(name="gather_gemv_cpu", version="0.0.1", ext_modules=ext_modules)
