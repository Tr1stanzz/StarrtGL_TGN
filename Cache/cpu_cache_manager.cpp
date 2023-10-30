#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <parallel_hashmap/phmap.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <ctime>


using namespace std;
namespace py = pybind11;
namespace th = torch;
typedef int64_t NodeIDType;

map<string,phmap::parallel_flat_hash_map <NodeIDType,NodeIDType> *> cache_index;
map<string,th::Tensor > cache_data;

class DataFromCPUCache
{
    public:
        th::Tensor data;
        th::Tensor cached_index;
        th::Tensor uncached_index;
        DataFromCPUCache(){}
        DataFromCPUCache(th::Tensor & _cached_index, th::Tensor & _uncached_index,
                           th::Tensor & _data):
                           data(_data), cached_index(_cached_index), uncached_index(_uncached_index){}
};



void cache_data2mem(string data_name,th::Tensor index,th::Tensor data){
    AT_ASSERTM(data.is_contiguous(), "Offset tensor must be contiguous");
    cache_data[data_name] = data;
    auto array = index.data_ptr<NodeIDType>();
    vector<pair<NodeIDType,NodeIDType>> v;
    int mem_size = data.size(0);
    for(int i=0;i<mem_size;i++){
        v.push_back(make_pair((NodeIDType)array[i],(NodeIDType)i));
        //cout<<(NodeIDType)array[i]<<" "<<(NodeIDType)i<<endl;
    }
    cache_index[data_name] = new phmap::parallel_flat_hash_map<NodeIDType,NodeIDType>(v.begin(),v.end());
}   
double tot1 = 0;
double tot2 = 0;
double tot3 = 0;
double tot4 = 0;
int cnt = 0;
DataFromCPUCache get_from_cache(string data_name,th::Tensor index){
    int len = index.size(0);
    auto array = index.data_ptr<NodeIDType>();

    phmap::parallel_flat_hash_map <NodeIDType,NodeIDType> * mp = cache_index[data_name];
    th::Tensor data = cache_data[data_name];
    vector<NodeIDType> iscached(len);
    //cout<<len<<endl;
#pragma omp parallel for num_threads(10)
    for(int i=0 ; i < len ; i++){
        NodeIDType id = (NodeIDType)array[i];
        
        if(mp->find(id) != mp->end()){
            iscached[i] = mp->find(id)->second;
            //cout<<i<<" "<<id<<" "<<mp->find(id)->second<<endl;
        }
        else{
            iscached[i] = -1;
            // cout<<i<<" "<<id<<" "<<-1<<endl;
        }
    }
    clock_t t0 = clock();
    th::Tensor is_cache = th::tensor(iscached);
    th::Tensor is_select = is_cache >= 0;
    clock_t t1 = clock();
    //cout<<is_cache.size(0)<<" "<<no_select.size(0)<<endl;
    th::Tensor cached_index = index.masked_select(is_select);
    th::Tensor uncache_index = index.masked_select(~is_select);
    clock_t t4 = clock();
    th::Tensor mem_index = is_cache.masked_select(is_select);
    clock_t t2 = clock();
    th::Tensor cached_data = data.index_select(0,mem_index);
    clock_t t3 = clock();
    DataFromCPUCache dataFromCache = DataFromCPUCache(cached_index,uncache_index,cached_data);
    cnt = cnt+1;
    tot1+=(double)(t1-t0)/CLOCKS_PER_SEC;
    tot2+=(double)(t4-t1)/CLOCKS_PER_SEC;
    tot3+=(double)(t2-t4)/CLOCKS_PER_SEC;
    tot4+=(double)(t3-t2)/CLOCKS_PER_SEC;
   // cout<<"cache"<<" "<<tot1/cnt<<" "<<tot2/cnt<<" "<<tot3/cnt<<" "<<tot4/cnt<<endl;
    return dataFromCache;
}
PYBIND11_MODULE(cpu_cache_manager, m)
{
    m
    .def("cache_data2mem", 
        &cache_data2mem, 
        py::return_value_policy::reference)
    .def("get_from_cache", 
        &get_from_cache, 
        py::return_value_policy::reference);
    py::class_<DataFromCPUCache>(m, "DataFromCPUCache")
    
        .def_readonly("cache_index", &DataFromCPUCache::cached_index, py::return_value_policy::reference)
        .def_readonly("uncache_index", &DataFromCPUCache::uncached_index,py::return_value_policy::reference) 
        .def_readonly("cache_data", &DataFromCPUCache::data, py::return_value_policy::reference);

}