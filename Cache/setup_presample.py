from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='presample_cores',
    ext_modules=[
        CppExtension(
            name='presample_cores',
            sources=['presample_cores.cpp'],
            extra_compile_args=['-fopenmp','-Xlinker',' -export-dynamic'],
            include_dirs=["./"],
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