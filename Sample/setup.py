from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='sample_cores',
    ext_modules=[
        CppExtension(
            name='sample_cores',
            sources=['sample_cores_dist.cpp'],
            extra_compile_args=['-fopenmp', '-Xlinker', ' -export-dynamic', '-O3', '-std=c++17'],
            include_dirs=["./parallel_hashmap"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
