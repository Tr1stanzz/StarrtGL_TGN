#!/bin/sh
#conda activate gnn
cd ./Sample
if [ -f "setup.py" ]; then
    rm -r build
    rm sample_cores.cpython-*.so
    python setup.py build_ext --inplace
fi
cd ../Cache
if [ -f "setup_manager.py" ]; then
    rm -r build
    rm cpu_cache_manager.cpython-*.so
    rm presample_cores.cpython-*.so
    python setup_manager.py build_ext --inplace
    python setup_presample.py build_ext --inplace
fi
cd ../part
if [  -f "setup.py" ]; then
    rm -r build
    rm torch_utils.cpython-*.so
    python setup.py build_ext --inplace
fi
cd ..
