#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <parallel_hashmap/phmap.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
using namespace std;
namespace py = pybind11;
namespace th = torch;
typedef int64_t NodeIDType;
th::Tensor sparse_get_index(th::Tensor in,th::Tensor map_key){
    auto key_ptr = map_key.data_ptr<NodeIDType>();
    auto in_ptr = in.data_ptr<NodeIDType>();
    int sz =  map_key.size(0);
    vector<pair<NodeIDType,NodeIDType>> mp(sz);
    vector<NodeIDType> out(in.size(0));
#pragma omp parallel for
    for(int i=0;i<sz;i++){
        mp[i] = make_pair(key_ptr[i],i);
    }
    phmap::parallel_flat_hash_map<NodeIDType,NodeIDType> dict(mp.begin(),mp.end());

#pragma omp parallel for
    for(int i=0;i<in.size(0);i++){
        out[i] = dict.find(in_ptr[i])->second;
    }
    return th::tensor(out);
}


PYBIND11_MODULE(torch_utils, m)
{
    m
    .def("sparse_get_index", 
        &sparse_get_index, 
        py::return_value_policy::reference);
}