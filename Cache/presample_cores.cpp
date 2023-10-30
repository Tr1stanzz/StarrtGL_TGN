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
//phmap::btree_map<string,phmap::btree_map<string,int>> node_freq;
map<string,map<NodeIDType,double> *> node_freq;
void update_count(string dataname,th::Tensor IdArray,char* device,double weight,NodeIDType l,NodeIDType r){
    //phmap::btree_map<NodeIDType,int> freq;
    auto array = IdArray.data_ptr<NodeIDType>();
    map<NodeIDType,double>* freq = nullptr;
    if(node_freq.find(dataname)!=node_freq.end()){
        freq  = node_freq[dataname];
    }
    else{
        cout<<"no this data "<<dataname<<endl;
        freq = new map<NodeIDType,double>();
        node_freq[dataname]=freq;
    }
    if(strcmp(device,"cpu")==0){
        for(int i=0;i<IdArray.size(0);i++){
            NodeIDType u = NodeIDType(array[i]);
            //cout<<u<<" "<<l<<" "<<r<<" "<<device<<endl;
            if(u<r and u>=l)continue;
            if(freq->find(u) != freq->end()){
                //cout<<u<<" ! "<<freq->find(u)->second + 1<<endl;
                freq->find(u)->second++;
                //freq->insert(std::make_pair(u,(freq->find(u))->second + 1));
            }
            else freq->insert(std::make_pair(u,1));
        };
    }
    else{
        for(int i=0;i<IdArray.size(0);i++){
            NodeIDType u = NodeIDType(array[i]);
            if(u<r and u>=l){
                if(freq->find(u) != freq->end())freq->insert(std::make_pair(u,(freq->find(u))->second + 1));
                else freq->insert(std::make_pair(u,1));
            }
            else{
                if(freq->find(u) != freq->end())freq->insert(std::make_pair(u,(freq->find(u))->second + weight));
                else freq->insert(std::make_pair(u,weight));
            }
        };
    }
    fflush(stdout);
}
typedef pair<NodeIDType,int> PAIR;
bool cmp(const PAIR &A, const PAIR &B){
    return A.second > B.second;
}
th::Tensor get_max_rank(string dataname,int size){
    map<NodeIDType,double>* nmap = node_freq[dataname];
    vector<PAIR> vec(nmap->begin(),nmap->end());
    sort(vec.begin(), vec.end(), cmp); 
    vector<NodeIDType> out;
    int vector_size = (int)vec.size();
    for(int i = 0; i < min(size,vector_size);i++){
        out.push_back(vec[i].first);
        //cout<<vec[i].first<<" "<<vec[i].second<<endl;
    }
    node_freq[dataname]->clear();
    delete node_freq[dataname];
    node_freq.erase(dataname);
    return th::tensor(out);
} 

PYBIND11_MODULE(presample_cores, m)
{
    m
    .def("update_count", 
        &update_count, 
        py::return_value_policy::reference)
    .def("get_max_rank", 
        &get_max_rank, 
        py::return_value_policy::reference);
}