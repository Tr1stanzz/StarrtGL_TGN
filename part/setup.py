from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torch_utils',
    ext_modules=[
        CppExtension(
            name='torch_utils',
            sources=['./torch_utils.cpp'],
            extra_compile_args=['-fopenmp','-Xlinker',' -export-dynamic'],
            include_dirs=["../Cache"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })#

#setup(
#    name='cpu_cache_manager',
#    ext_modules=[
#        CppExtension(
#            name='cpu_cache_manager',
#            sources=['cpu_cache_manager.cpp'],
#            extra_compile_args=['-fopenmp','-Xlinker',' -export-dynamic'],
#            include_dirs=["./"],
#        ),
#    ],
#    cmdclass={
#        'build_ext': BuildExtension
#    })#
#