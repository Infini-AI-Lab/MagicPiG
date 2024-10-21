from torch.utils.cpp_extension import CppExtension
import setuptools

ext_modules = [
    CppExtension(
        "lsh",
        sources=[
            "lsh.cc",
        ],
        extra_compile_args=["-mavx512f", "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0","-std=c++17"],
        extra_link_args=['-fopenmp'],
    ),
]

setuptools.setup(name="lsh", version="0.0.1", ext_modules=ext_modules)
