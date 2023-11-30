#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <time.h>
#include <random>
#include <phmap.h>
#include <boost/thread/mutex.hpp>
#define MTX boost::mutex

using namespace std;
namespace py = pybind11;
namespace th = torch;

typedef int64_t NodeIDType;
typedef int64_t EdgeIDType;
typedef float WeightType;
typedef float TimeStampType;


#define EXTRAARGS , phmap::priv::hash_default_hash<K>, \
                            phmap::priv::hash_default_eq<K>, \
                            std::allocator<K>, 4, MTX
template <class K>
using HashT  = phmap::parallel_flat_hash_set<K EXTRAARGS>;

template <class K, class V>
using HashM  = phmap::parallel_flat_hash_map<K, V EXTRAARGS>;


class TemporalNeighborBlock;
class TemporalGraphBlock;
TemporalNeighborBlock& get_neighbors(th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time);
vector<th::Tensor> neighbor_sample_from_nodes(
    th::Tensor nodes, TemporalNeighborBlock& tnb, 
    int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, 
    string policy, optional<th::Tensor> root_ts, optional<int> is_root_ts, 
    optional<TimeStampType> start, optional<TimeStampType> end);
vector<th::Tensor> neighbor_sample_from_nodes_static(th::Tensor nodes, TemporalNeighborBlock& tnb, int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, string policy);
vector<th::Tensor> neighbor_sample_from_nodes_with_time(th::Tensor nodes, TimeStampType start, TimeStampType end, TemporalNeighborBlock& tnb, int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, string policy);
vector<th::Tensor> neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts, TemporalNeighborBlock& tnb, int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, string policy, int is_root_ts);
th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads);
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr);
int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr);
vector<th::Tensor> divide_nodes_to_part(th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads);
// vector<int64_t> sample_multinomial(vector<WeightType> weights, int num_samples, bool replacement, default_random_engine e);
NodeIDType sample_multinomial(const vector<WeightType>& weights, default_random_engine& e);


template<typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    // non-copy value transfer
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

/* 
 * NeighborSampler Utils
 */
class TemporalNeighborBlock
{
    public:
        vector<vector<NodeIDType>> neighbors;
        vector<vector<TimeStampType>> timestamp;
        vector<vector<EdgeIDType>> eid;
        vector<vector<WeightType>> edge_weight;
        vector<phmap::parallel_flat_hash_map<NodeIDType, int64_t>> inverted_index;
        vector<int64_t> deg;
        vector<phmap::parallel_flat_hash_set<NodeIDType>> neighbors_set;

        bool with_eid = false;
        bool weighted = false;
        bool with_timestamp = false;

        TemporalNeighborBlock(){}
        // TemporalNeighborBlock(const TemporalNeighborBlock &tnb);
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), deg(deg){}
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight, 
                              vector<vector<EdgeIDType>>& eid,
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight),eid(eid), deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; }
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight,
                              vector<vector<TimeStampType>>& timestamp,
                              vector<vector<EdgeIDType>>& eid, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight), timestamp(timestamp),eid(eid), deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; 
                                this->with_timestamp=true;}
        py::array get_node_neighbor(NodeIDType node_id){
            return vec2npy(neighbors[node_id]);
        }
        py::array get_node_neighbor_timestamp(NodeIDType node_id){
            return vec2npy(timestamp[node_id]);
        }
        int64_t get_node_deg(NodeIDType node_id){
            return deg[node_id];
        }
        bool empty(){
            return this->deg.empty();
        }
        void update_edge_weight(TemporalNeighborBlock& tnb, th::Tensor row, th::Tensor col, th::Tensor edge_weight);
        int64_t update_neighbors_with_time(TemporalNeighborBlock& tnb, th::Tensor row, th::Tensor col, th::Tensor time, th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight);

        
        std::string serialize() const {
            std::ostringstream oss;
            // 序列化基本类型成员
            oss << with_eid << " " << weighted << " " << with_timestamp << " ";

            // 序列化 vector<vector<T>> 类型成员
            auto serializeVecVec = [&oss](const auto& vecVec) {
                for (const auto& vec : vecVec) {
                    oss << vec.size() << " ";
                    for (const auto& elem : vec) {
                        oss << elem << " ";
                    }
                }
                oss << "|";  // 添加一个分隔符以区分不同的 vector
            };

            serializeVecVec(neighbors);
            serializeVecVec(timestamp);
            serializeVecVec(eid);
            serializeVecVec(edge_weight);

            // 序列化 vector<int64_t> 类型成员
            oss << deg.size() << " ";
            for (const auto& d : deg) {
                oss << d << " ";
            }
            oss << "|";

            // 序列化 inverted_index
            for (const auto& map : inverted_index) {
                oss << map.size() << " ";
                for (const auto& [key, value] : map) {
                    oss << key << " " << value << " ";
                }
            }
            oss << "|";

            // 序列化 neighbors_set
            for (const auto& set : neighbors_set) {
                oss << set.size() << " ";
                for (const auto& elem : set) {
                    oss << elem << " ";
                }
            }
            oss << "|";

            return oss.str();
        }

        static TemporalNeighborBlock deserialize(const std::string& s) {
            std::istringstream iss(s);
            TemporalNeighborBlock tnb;

            // 反序列化基本类型成员
            iss >> tnb.with_eid >> tnb.weighted >> tnb.with_timestamp;

            // 反序列化 vector<vector<T>> 类型成员
            auto deserializeVecLong = [&iss](vector<vector<int64_t>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break;  // 防止多余的空白
                    vector<int64_t> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            
            auto deserializeVecFloat = [&iss](vector<vector<float>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break;  // 防止多余的空白
                    vector<float> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            deserializeVecLong(tnb.neighbors);
            deserializeVecFloat(tnb.timestamp);
            deserializeVecLong(tnb.eid);
            deserializeVecFloat(tnb.edge_weight);

            std::string segment;
            // 反序列化 vector<int64_t> 类型成员
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream vec_iss(segment);
            size_t vec_size;
            vec_iss >> vec_size;
            tnb.deg.resize(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                vec_iss >> tnb.deg[i];
            }

            // 反序列化 inverted_index
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream map_iss(segment);
            while (!map_iss.eof()) {
                size_t map_size;
                map_iss >> map_size;
                if (map_iss.eof()) break;
                phmap::parallel_flat_hash_map<NodeIDType, int64_t> map;
                for (size_t i = 0; i < map_size; ++i) {
                    NodeIDType key;
                    int64_t value;
                    map_iss >> key >> value;
                    map[key] = value;
                }
                tnb.inverted_index.push_back(map);
            }

            // 反序列化 neighbors_set
            std::getline(iss, segment, '|');
            std::istringstream set_iss(segment);
            while (!set_iss.eof()) {
                size_t set_size;
                set_iss >> set_size;
                if (set_iss.eof()) break;
                phmap::parallel_flat_hash_set<NodeIDType> set;
                for (size_t i = 0; i < set_size; ++i) {
                    NodeIDType elem;
                    set_iss >> elem;
                    set.insert(elem);
                }
                tnb.neighbors_set.push_back(set);
            }

            return tnb;
        }
};

class TemporalGraphBlock
{
    public:
        vector<NodeIDType> row;
        vector<NodeIDType> col;
        vector<EdgeIDType> eid;
        vector<NodeIDType> nid;
        vector<TimeStampType> e_ts;
        vector<TimeStampType> src_ts;
        vector<NodeIDType> sample_nodes;
        vector<TimeStampType> sample_nodes_ts;
        vector<TimeStampType> sample_nodes_dts;

        TemporalGraphBlock(){}
        // TemporalGraphBlock(const TemporalGraphBlock &tgb);
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes,
                           vector<TimeStampType> &_sample_nodes_ts,
                           vector<TimeStampType> &_sample_nodes_dts):
                           row(_row), col(_col), sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts), 
                           sample_nodes_dts(_sample_nodes_dts){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                            vector<EdgeIDType> &_eid,vector<EdgeIDType> &_nid, vector<TimeStampType> &_e_ts,vector<TimeStampType> &_src_ts,
                           vector<NodeIDType> &_sample_nodes,
                           vector<TimeStampType> &_sample_nodes_ts,
                           vector<TimeStampType> &_sample_nodes_dts):
                           row(_row), col(_col), eid(_eid),nid(_nid),e_ts(_e_ts),src_ts(_src_ts),sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts), 
                           sample_nodes_dts(_sample_nodes_dts){}
};

// 辅助函数
template <typename T>
T* get_data_ptr(const th::Tensor& tensor) {
    AT_ASSERTM(tensor.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(tensor.dim() == 1, "Offset tensor must be one-dimensional");
    return tensor.data_ptr<T>();
}

TemporalNeighborBlock& get_neighbors(
        th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time)
{   //row、col、time按time升序排列，由时间早的到时间晚的
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    EdgeIDType* eid_ptr = eid ? get_data_ptr<EdgeIDType>(eid.value()) : nullptr;
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    TimeStampType* t = time ? get_data_ptr<TimeStampType>(time.value()) : nullptr;
    int64_t edge_num = row.size(0);
    static TemporalNeighborBlock tnb = TemporalNeighborBlock();

    double start_time = omp_get_wtime();
    //初始化
    tnb.neighbors.resize(num_nodes);
    tnb.deg.resize(num_nodes, 0);
        //初始化optional相关
    tnb.with_eid = eid.has_value();
    tnb.weighted = edge_weight.has_value();
    tnb.with_timestamp = time.has_value();
    if (tnb.with_eid) tnb.eid.resize(num_nodes);
    if (tnb.weighted) {
        tnb.edge_weight.resize(num_nodes);
        tnb.inverted_index.resize(num_nodes);
    }
    if (tnb.with_timestamp) tnb.timestamp.resize(num_nodes);
        
    //计算, 条件判断移出循环优化执行效率
    for(int64_t i=0; i<edge_num; i++){
        //计算节点邻居
        tnb.neighbors[dst[i]].emplace_back(src[i]);
    }
    //如果有eid，插入
    if(tnb.with_eid)
        for(int64_t i=0; i<edge_num; i++){
            tnb.eid[dst[i]].emplace_back(eid_ptr[i]);
        }
    //如果有权重信息，插入节点与邻居边的权重和反向索引
    if(tnb.weighted)
        for(int64_t i=0; i<edge_num; i++){
            tnb.edge_weight[dst[i]].emplace_back(ew[i]);
            if(tnb.with_eid) tnb.inverted_index[dst[i]][eid_ptr[i]]=tnb.neighbors[dst[i]].size()-1;
            else tnb.inverted_index[dst[i]][src[i]]=tnb.neighbors[dst[i]].size()-1;
        }
    //如果有时序信息，插入节点与邻居边的时间
    if(tnb.with_timestamp)
        for(int64_t i=0; i<edge_num; i++){
            tnb.timestamp[dst[i]].emplace_back(t[i]);
        }
        
    if(is_distinct){
        for(int64_t i=0; i<num_nodes; i++){
            //收集单边去重节点度
            phmap::parallel_flat_hash_set<NodeIDType> temp_s;
            temp_s.insert(tnb.neighbors[i].begin(), tnb.neighbors[i].end());
            tnb.neighbors_set.emplace_back(temp_s);
            tnb.deg[i] = tnb.neighbors_set[i].size();
        }
    }
    else{
        for(int64_t i=0; i<num_nodes; i++){
            //收集单边节点度
            tnb.deg[i] = tnb.neighbors[i].size();
        }
    }
    double end_time = omp_get_wtime();
    cout<<"get_neighbors consume: "<<end_time-start_time<<"s"<<endl;
    return tnb;
}

void TemporalNeighborBlock::update_edge_weight(
        TemporalNeighborBlock& tnb, th::Tensor row_or_eid, th::Tensor col, th::Tensor edge_weight){
    AT_ASSERTM(tnb.weighted, "This Graph has no edge weight infomation");
    auto dst = get_data_ptr<NodeIDType>(col);
    WeightType* ew = get_data_ptr<WeightType>(edge_weight);
    NodeIDType* src;
    EdgeIDType* eid_ptr;
    if(tnb.with_eid) eid_ptr = get_data_ptr<EdgeIDType>(row_or_eid);
    else src = get_data_ptr<NodeIDType>(row_or_eid);
    
    int64_t edge_num = col.size(0);

    for(int64_t i=0; i<edge_num; i++){
        //修改节点与邻居边的权重
        AT_ASSERTM(tnb.inverted_index[dst[i]].count(src[i])==1, "Unexist Edge Index: "+to_string(src[i])+", "+to_string(dst[i]));
		int index;
        if(tnb.with_eid) index = tnb.inverted_index[dst[i]][eid_ptr[i]];
        else index = tnb.inverted_index[dst[i]][src[i]];
        tnb.edge_weight[dst[i]][index] = ew[i];
    }
}

int64_t TemporalNeighborBlock::update_neighbors_with_time(
        TemporalNeighborBlock& tnb, th::Tensor row, th::Tensor col, th::Tensor time,th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight){
        //row、col、time按time升序排列，由时间早的到时间晚的
    AT_ASSERTM(tnb.empty(), "Empty TemporalNeighborBlock, please use get_neighbors_with_time");
    AT_ASSERTM(tnb.with_timestamp == true, "This Graph has no time infomation!");
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    auto eid_ptr = get_data_ptr<EdgeIDType>(eid);
    auto t = get_data_ptr<TimeStampType>(time);
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    int64_t edge_num = row.size(0);
    int64_t num_nodes = tnb.neighbors.size();

    //处理optional的值
    if(edge_weight.has_value()){
        AT_ASSERTM(tnb.weighted == true, "This Graph has no edge weight");
    }
    if(tnb.weighted){
        AT_ASSERTM(edge_weight.has_value(), "This Graph need edge weight");
    }

    // double start_time = omp_get_wtime();
    if(is_distinct){
        for(int64_t i=0; i<edge_num; i++){
            //如果有新节点
            if(dst[i]>=num_nodes){
                num_nodes = dst[i]+1;
                tnb.neighbors.resize(num_nodes);
                tnb.deg.resize(num_nodes, 0);
                tnb.eid.resize(num_nodes);
                tnb.timestamp.resize(num_nodes);
                    //初始化optional相关
                if (tnb.weighted) {
                    tnb.edge_weight.resize(num_nodes);
                    tnb.inverted_index.resize(num_nodes);
                }
            }
            //更新节点邻居
            tnb.neighbors[dst[i]].emplace_back(src[i]);
            //插入eid
            tnb.eid[dst[i]].emplace_back(eid_ptr[i]);
            //插入节点与邻居边的时间
            tnb.timestamp[dst[i]].emplace_back(t[i]);
            //如果有权重信息，插入节点与邻居边的权重和反向索引
            if(tnb.weighted){
                tnb.edge_weight[dst[i]].emplace_back(ew[i]);
                if(tnb.with_eid) tnb.inverted_index[dst[i]][eid_ptr[i]]=tnb.neighbors[dst[i]].size()-1;
                else tnb.inverted_index[dst[i]][src[i]]=tnb.neighbors[dst[i]].size()-1;
            }
            
            tnb.neighbors_set[dst[i]].insert(src[i]);
            tnb.deg[dst[i]]=tnb.neighbors_set[dst[i]].size();
        }
    }
    else{
        for(int64_t i=0; i<edge_num; i++){
            //更新节点邻居
            tnb.neighbors[dst[i]].emplace_back(src[i]);
            //插入eid
            tnb.eid[dst[i]].emplace_back(eid_ptr[i]);
            //插入节点与邻居边的时间
            tnb.timestamp[dst[i]].emplace_back(t[i]);
            //如果有权重信息，插入节点与邻居边的权重和反向索引
            if(tnb.weighted){
                tnb.edge_weight[dst[i]].emplace_back(ew[i]);
                tnb.inverted_index[dst[i]][src[i]]=tnb.neighbors[dst[i]].size()-1;
            }

            tnb.deg[dst[i]]=tnb.neighbors[dst[i]].size();
        }
    }
    // double end_time = omp_get_wtime();
    // cout<<"update_neighbors consume: "<<end_time-start_time<<"s"<<endl;
    return num_nodes;
}

vector<th::Tensor> neighbor_sample_from_nodes(
    th::Tensor nodes, TemporalNeighborBlock& tnb, 
    int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, 
    string policy, optional<th::Tensor> root_ts, optional<int> is_root_ts, 
    optional<TimeStampType> start, optional<TimeStampType> end)
{
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform")
        ;
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        if(start.has_value()){
            AT_ASSERTM(end.has_value(), "Parameter mismatch!");
            return neighbor_sample_from_nodes_with_time(nodes, start.value(), end.value(), tnb, pid, part_ptr, fanout, threads, policy);
        }
        else{
            AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
            AT_ASSERTM(is_root_ts.has_value(), "Parameter mismatch!");
            return neighbor_sample_from_nodes_with_before(nodes, root_ts.value(), tnb, pid, part_ptr, fanout, threads, policy, is_root_ts.value());
        }
    }
    else{
        return neighbor_sample_from_nodes_static(nodes, tnb, pid, part_ptr, fanout, threads, policy);
    }
}

vector<th::Tensor> neighbor_sample_from_nodes_static(
        th::Tensor nodes, TemporalNeighborBlock& tnb, 
        int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, string policy){
    py::gil_scoped_release release;
    TemporalGraphBlock tgb = TemporalGraphBlock();
    vector<th::Tensor> ret(4);
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);    
    vector<phmap::parallel_flat_hash_set<NodeIDType>> node_s_threads(threads);
    phmap::parallel_flat_hash_set<NodeIDType> node_s;
    vector<vector<NodeIDType>> row_threads(threads), col_threads(threads), eid_threads(threads);
    bool with_eid = tnb.with_eid;
    
    double start_time = omp_get_wtime();
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        vector<NodeIDType> nei(tnb.neighbors[node]);
        vector<EdgeIDType> edge;
        if(with_eid) edge = tnb.eid[node];
        TemporalGraphBlock tgb_i = TemporalGraphBlock();

        if(tnb.deg[node]>fanout){
            // //度大于扇出的话需要随机选择fanout个邻居
            // if(tnb.weighted){//考虑边权重信息
            //     const vector<WeightType>& ew = tnb.edge_weight[node];
            //     bool replacement = false;
            //     default_random_engine e(time(0));
            //     vector<NodeIDType> indices  = sample_multinomial(ew, fanout, replacement, e);
            //     for(int index : indices){
            //         NodeIDType nid = nei[index];
            //         if(with_eid){
            //             EdgeIDType eid = edge[index];
            //             tgb_i.eid.emplace_back(eid);
            //         }
            //         node_s_threads[tid].insert(nid);
            //         tgb_i.row.emplace_back(nid);
            //     }
            // }
            // else{//没有权重信息
            //     phmap::flat_hash_set<NodeIDType> temp_s;
            //     default_random_engine e(time(0));
            //     uniform_int_distribution<> u(0, tnb.deg[node]-1);
            //     while(temp_s.size()!=fanout){
            //         //循环选择fanout个邻居
            //         auto rnd = u(e);
            //         auto chosen_n_iter = nei.begin() + rnd;
            //         auto rst = temp_s.insert(*chosen_n_iter);
            //         if(rst.second){ //不重复
            //             if(with_eid){
            //                 auto chosen_e_iter = edge.begin() + rnd;
            //                 tgb_i.eid.emplace_back(*chosen_e_iter);
            //             }
            //             node_s_threads[tid].insert(*chosen_n_iter);
            //         }
            //     }
            //     tgb_i.row.assign(temp_s.begin(), temp_s.end());
            // }
            phmap::flat_hash_set<NodeIDType> temp_s;
            default_random_engine e(time(0));
            uniform_int_distribution<> u(0, tnb.deg[node]-1);
            while(temp_s.size()!=fanout){
                //循环选择fanout个邻居
                NodeIDType indice;
                if(policy == "weighted"){//考虑边权重信息
                    // cout<<"weighted sample"<<endl;
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    indice = sample_multinomial(ew, e);
                }
                else if(policy == "uniform"){//均匀采样
                    // cout<<"uniform sample"<<endl;
                    indice = u(e);
                }
                auto chosen_n_iter = nei.begin() + indice;
                auto rst = temp_s.insert(*chosen_n_iter);
                if(rst.second){ //不重复
                    if(with_eid){
                        auto chosen_e_iter = edge.begin() + indice;
                        tgb_i.eid.emplace_back(*chosen_e_iter);
                    }
                    node_s_threads[tid].insert(*chosen_n_iter);
                }
            }
            tgb_i.row.assign(temp_s.begin(), temp_s.end());
        }
        else{
            node_s_threads[tid].insert(nei.begin(), nei.end());
            tgb_i.row.swap(nei);
            if(with_eid) tgb_i.eid.swap(edge);
        }
        tgb_i.col.resize(tgb_i.row.size(), node);
        row_threads[tid].insert(row_threads[tid].end(),tgb_i.row.begin(),tgb_i.row.end());
        col_threads[tid].insert(col_threads[tid].end(),tgb_i.col.begin(),tgb_i.col.end());
        if(with_eid) eid_threads[tid].insert(eid_threads[tid].end(),tgb_i.eid.begin(),tgb_i.eid.end());
    }
    double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
#pragma omp sections
{
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(row_threads[i].data(), {row_threads[i].size()}, torch::kInt64);
        ret[0] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(col_threads[i].data(), {col_threads[i].size()}, torch::kInt64);
        ret[1] = (torch::cat(tv));
    }
#pragma omp section
    {
        if(with_eid)
        {
            vector<th::Tensor> tv(threads);
            for(int i = 0; i<threads; i++)
                tv[i] = th::from_blob(eid_threads[i].data(), {eid_threads[i].size()}, torch::kInt64);
            ret[2] = (torch::cat(tv));
        }
    }
#pragma omp section
    // {
    //     vector<th::Tensor> tv(threads);
    //     vector<vector<NodeIDType>> n(threads);
    //     for(int i = 0; i<threads; i++){
    //         n[i].assign(node_s_threads[i].begin(), node_s_threads[i].end());
    //         tv[i] = th::from_blob(n[i].data(), {n[i].size()});
    //     }
    //     ret[3] = (torch::cat(tv));
    // }
        for(int i = 0; i<threads; i++)
            node_s.insert(node_s_threads[i].begin(), node_s_threads[i].end());
}
    //sampled nodes 去重, 暂不插入root nodes
    start_time = end_time;
    tgb.sample_nodes.assign(node_s.begin(), node_s.end());
    end_time = omp_get_wtime();
    // cout<<"end unique consume: "<<end_time-start_time<<"s"<<endl;

    //出发点在前，到达点在后
    // ret[0] = th::tensor(tgb.row);
    // ret[1] = th::tensor(tgb.col);
    // if(with_eid) ret[2] = th::tensor(tgb.eid);
    ret[3] = th::tensor(tgb.sample_nodes);

    py::gil_scoped_acquire acquire;
    return ret;
}

vector<th::Tensor> neighbor_sample_from_nodes_with_time(
        th::Tensor nodes, TimeStampType start, TimeStampType end, TemporalNeighborBlock& tnb, 
        int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads, string policy){
    py::gil_scoped_release release;
    TemporalGraphBlock tgb = TemporalGraphBlock();
    vector<th::Tensor> ret(3);
    AT_ASSERTM(nodes.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(nodes.dim() == 1, "0ffset tensor must be one-dimensional");
    auto nodes_data = nodes.data_ptr<NodeIDType>();
    vector<phmap::parallel_flat_hash_set<NodeIDType>> node_s_threads(threads);
    phmap::parallel_flat_hash_set<NodeIDType> node_s;
    vector<vector<NodeIDType>> row_threads(threads), col_threads(threads);
    
    default_random_engine e(time(0));
    double start_time = omp_get_wtime();
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TemporalGraphBlock tgb_i = TemporalGraphBlock();
        int64_t start_index, end_index;
        if(start<0) start_index=0;
        else start_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), start)-tnb.timestamp[node].begin();
        end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), end)-tnb.timestamp[node].begin();
        if(tnb.deg[node]>fanout&&end_index-start_index>fanout){
            //度大于扇出的话需要随机选择fanout个邻居
            phmap::flat_hash_set<NodeIDType> temp_s;
            uniform_int_distribution<> u(start_index, end_index-1);
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            while(temp_s.size()!=fanout){
                //循环选择fanout个邻居
                // auto rand = u(e);
                // auto chosen_iter = tnb.neighbors[node].begin() + rand;
                // auto ts_iter = tnb.timestamp[node].begin() + rand;
                // if(*ts_iter<start)
                //     continue;
                // else if(*ts_iter>end)
                //     continue;
                NodeIDType indice;
                if(policy == "weighted"){//考虑边权重信息
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    indice = sample_multinomial(ew, e);
                }
                else if(policy == "uniform"){//均匀采样
                    indice = u(e);
                }
                auto chosen_iter = tnb.neighbors[node].begin() + indice;
                node_s_threads[tid].insert(*chosen_iter);
                temp_s.insert(*chosen_iter);
            }
            tgb_i.row.assign(temp_s.begin(), temp_s.end());
        }
        else{
            phmap::flat_hash_set<NodeIDType> temp_s(tnb.neighbors[node].begin()+start_index, tnb.neighbors[node].begin()+end_index);
            node_s_threads[tid].insert(temp_s.begin(), temp_s.end());
            tgb_i.row.assign(temp_s.begin(), temp_s.end());
        }
        tgb_i.col.resize(tgb_i.row.size(), node);
        row_threads[tid].insert(row_threads[tid].end(),tgb_i.row.begin(),tgb_i.row.end());
        col_threads[tid].insert(col_threads[tid].end(),tgb_i.col.begin(),tgb_i.col.end());
    }
    double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;

#pragma omp sections
{
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(row_threads[i].data(), {row_threads[i].size()}, torch::kInt64);
        ret[0] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(col_threads[i].data(), {col_threads[i].size()}, torch::kInt64);
        ret[1] = (torch::cat(tv));
    }
#pragma omp section
    {
        size_t total_size = 0;
        for (const auto& node_s : node_s_threads) {
            total_size += node_s.size();
        }
        node_s.reserve(total_size);
        for(int i = 0; i<threads; i++)
            node_s.insert(node_s_threads[i].begin(), node_s_threads[i].end());
    }
}
    //sampled nodes 去重, 暂不插入root nodes
    start_time = end_time;
    tgb.sample_nodes.assign(node_s.begin(), node_s.end());
    end_time = omp_get_wtime();
    // cout<<"end unique consume: "<<end_time-start_time<<"s"<<endl;

    ret[2] = th::tensor(tgb.sample_nodes);

    py::gil_scoped_acquire acquire;
    return ret;
}

vector<th::Tensor> neighbor_sample_from_nodes_with_before(
        th::Tensor nodes, th::Tensor root_ts,TemporalNeighborBlock& tnb, 
        int pid, const vector<NodeIDType>& part_ptr, int64_t fanout, int threads,string policy,int is_root_ts){
    py::gil_scoped_release release;
    TemporalGraphBlock tgb = TemporalGraphBlock();
    vector<th::Tensor> ret(7);
    AT_ASSERTM(nodes.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(nodes.dim() == 1, "0ffset tensor must be one-dimensional");
    auto nodes_data = nodes.data_ptr<NodeIDType>();
    auto ts_data = root_ts.data_ptr<TimeStampType>();
    HashT<pair<NodeIDType,TimeStampType> > node_s;
    vector<TemporalGraphBlock> tgb_i(threads);
    
    default_random_engine e(time(0));
    double start_time = omp_get_wtime();
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
        // int end_index = tnb.timestamp[node].end()-tnb.timestamp[node].begin();
        //cout<<node<<" "<<end_index<<" "<<tnb.deg[node]<<endl;
        if(tnb.deg[node]>fanout&&end_index>fanout){
            //度大于扇出的话需要随机选择fanout个邻居
            phmap::flat_hash_set<NodeIDType> temp_s;
            uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            while(temp_s.size()!=fanout){
                //循环选择fanout个邻居
                int cid = 0;
                if(policy == "uniform"){
                    // cout<<"uniform sample"<<endl;
                    int rand = u(e);
                    if(temp_s.find(rand) != temp_s.end()) continue;
                    cid = rand;
                }
                else if(policy == "weighted"){
                    // cout<<"weighted sample"<<endl;
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    NodeIDType indice = sample_multinomial(ew, e);               
                    if(temp_s.find(indice) != temp_s.end()) continue;
                    cid = indice;
                }
                else if(policy == "recent"){
                    // cout<<"recent sample"<<endl;
                    cid = --end_index;
                }
                if(nodeIdToInOut(tnb.neighbors[node][cid], pid, part_ptr)==0)
                    if(is_root_ts){
                        node_s.emplace(make_pair(tnb.neighbors[node][cid],rtts));
                    }
                    else {
                        node_s.emplace(make_pair(tnb.neighbors[node][cid],tnb.timestamp[node][cid]));
                    }
                else
                    if(is_root_ts){
                        node_s.emplace(make_pair(tnb.neighbors[node][cid],rtts));
                    }
                    else {
                        node_s.emplace(make_pair(tnb.neighbors[node][cid],tnb.timestamp[node][cid]));
                    }
                temp_s.insert(cid);
                //cout<<"insert cid"<<endl;
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
                tgb_i[tid].e_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].row.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].col.emplace_back(node);
                tgb_i[tid].src_ts.emplace_back(rtts);
                //cout<<"cid"<<cid<<endl;
            }            
        }
        else{
            for(int cid = 0; cid < end_index;cid++){
                if(is_root_ts)node_s.emplace(make_pair(tnb.neighbors[node][cid],rtts));
                else node_s.emplace(make_pair(tnb.neighbors[node][cid],tnb.timestamp[node][cid]));
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
                tgb_i[tid].e_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].src_ts.emplace_back(rtts);
                tgb_i[tid].row.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].col.emplace_back(node);
            }
        }
       
    }
    double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    start_time = omp_get_wtime();

    tgb.sample_nodes.resize(node_s.size());
    tgb.sample_nodes_ts.resize(node_s.size());

    
#pragma omp sections
{
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(tgb_i[i].row.data(), {tgb_i[i].row.size()}, torch::kInt64);
        ret[0] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(tgb_i[i].col.data(), {tgb_i[i].col.size()}, torch::kInt64);
        ret[1] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(tgb_i[i].eid.data(), {tgb_i[i].eid.size()}, torch::kInt64);
        ret[2] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(tgb_i[i].e_ts.data(), {tgb_i[i].e_ts.size()}, torch::kFloat);
        ret[3] = (torch::cat(tv));
    }
#pragma omp section
    {
        vector<th::Tensor> tv(threads);
        for(int i = 0; i<threads; i++)
            tv[i] = th::from_blob(tgb_i[i].src_ts.data(), {tgb_i[i].src_ts.size()}, torch::kFloat);
        ret[4] = (torch::cat(tv));
    }
#pragma omp section
    {
            int i = 0;
        for(auto &e : node_s ){
            tgb.sample_nodes[i] = e.first;
            tgb.sample_nodes_ts[i] = e.second;
            i++;
        }
    }

}
            
    //sampled nodes 去重, 暂不插入root nodes
    // phmap::parallel_flat_hash_set<NodeIDType> s(tgb.col.begin(), tgb.col.end());
    end_time = omp_get_wtime();
    // cout<<"end unique consume: "<<end_time-start_time<<"s"<<endl;

    start_time = omp_get_wtime();

    // ret[0] = (th::tensor(tgb.row));
    // ret[1] = (th::tensor(tgb.col));
    // ret[2] = (th::tensor(tgb.eid));
    // ret[3] = (th::tensor(tgb.e_ts));
    // ret[4] = (th::tensor(tgb.src_ts));
    ret[5] = (th::tensor(tgb.sample_nodes));
    ret[6] = (th::tensor(tgb.sample_nodes_ts));
    
    end_time = omp_get_wtime();
    // cout<<"end trans consume: "<<end_time-start_time<<"s"<<endl;

    py::gil_scoped_acquire acquire;
    return ret;
}

/*-------------------------------------------------------------------------------------**
**------------Utils--------------------------------------------------------------------**
**-------------------------------------------------------------------------------------*/

th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads){
    auto array_ptr = array.data_ptr<NodeIDType>();
    phmap::parallel_flat_hash_set<NodeIDType> s(array_ptr, array_ptr+array.numel());
    if(heads.numel()==0) return th::tensor(vector<NodeIDType>(s.begin(), s.end()));
    AT_ASSERTM(heads.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(heads.dim() == 1, "0ffset tensor must be one-dimensional");
    auto heads_ptr = heads.data_ptr<NodeIDType>();
#pragma omp parallel for num_threads(threads)
    for(int64_t i=0; i<heads.size(0); i++){
        if(s.count(heads_ptr[i])==1){
        #pragma omp critical(erase)
            s.erase(heads_ptr[i]);
        }
    }
    vector<NodeIDType> ret;
    ret.reserve(s.size()+heads.numel());
    ret.assign(heads_ptr, heads_ptr+heads.numel());
    ret.insert(ret.end(), s.begin(), s.end());
    // cout<<"s: "<<s.size()<<" array: "<<array.size()<<endl;
    return th::tensor(ret);
}

int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr){
    int partitionId = -1;
    for(int i=0;i<part_ptr.size()-1;i++){
            if(nid>=part_ptr[i]&&nid<part_ptr[i+1]){
                partitionId = i;
                break;
            }
    }
    if(partitionId<0) throw "nid 不存在对应的分区";
    return partitionId;
}
//0:inner; 1:outer
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr){
    if(nid>=part_ptr[pid]&&nid<part_ptr[pid+1]){
        return 0;
    }
    return 1;
}

vector<th::Tensor> divide_nodes_to_part(
        th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads){
    double start_time = omp_get_wtime();
    AT_ASSERTM(nodes.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(nodes.dim() == 1, "0ffset tensor must be one-dimensional");
    auto nodes_id = nodes.data_ptr<NodeIDType>();
    vector<vector<vector<NodeIDType>>> node_part_threads;
    vector<th::Tensor> result(part_ptr.size()-1);
    //初始化点的分区，每个分区按线程划分避免冲突
    for(int i = 0; i<threads; i++){
        vector<vector<NodeIDType>> node_parts;
        for(int j=0;j<part_ptr.size()-1;j++){
            node_parts.push_back(vector<NodeIDType>());
        }
        node_part_threads.push_back(node_parts);
    }
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        int pid = nodeIdToPartId(nodes_id[i], part_ptr);
        node_part_threads[tid][pid].emplace_back(nodes_id[i]);
    }
#pragma omp parallel for num_threads(part_ptr.size()-1) default(shared)
    for(int i = 0; i<part_ptr.size()-1; i++){
        vector<NodeIDType> temp;
        for(int j=0;j<threads;j++){
            temp.insert(temp.end(), node_part_threads[j][i].begin(), node_part_threads[j][i].end());
        }
        result[i]=th::tensor(temp);
    }
    double end_time = omp_get_wtime();
    // cout<<"end divide consume: "<<end_time-start_time<<"s"<<endl;
    return result;
}

NodeIDType sample_multinomial(const vector<WeightType>& weights, default_random_engine& e){
    NodeIDType sample_indice;
    vector<WeightType> cumulative_weights;
    partial_sum(weights.begin(), weights.end(), back_inserter(cumulative_weights));
    AT_ASSERTM(cumulative_weights.back() > 0, "Edge weight sum should be greater than 0.");
    
    uniform_real_distribution<WeightType> distribution(0.0, cumulative_weights.back());
    WeightType random_value = distribution(e);
    auto it = lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
    sample_indice = distance(cumulative_weights.begin(), it);
    return sample_indice;
}

// vector<int64_t> sample_multinomial(vector<WeightType> weights, int num_samples, bool replacement, default_random_engine e) {
//     vector<int64_t> sample_indices;
//     vector<WeightType> cumulative_weights;
//     partial_sum(weights.begin(), weights.end(), back_inserter(cumulative_weights));
//     AT_ASSERTM(cumulative_weights.back() > 0, "Edge weight sum should be greater than 0.");

//     // default_random_engine e(time(0));
//     for (int i = 0; i < num_samples; ++i) {        
//         uniform_real_distribution<WeightType> distribution(0.0, cumulative_weights.back());
//         WeightType random_value = distribution(e);
//         auto it = lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
//         int64_t index = distance(cumulative_weights.begin(), it);
//         sample_indices.emplace_back(index);

//         if (!replacement) {
//             // Remove selected item and update cumulative_weights
//             weights.erase(weights.begin() + index);
//             cumulative_weights.clear();
//             partial_sum(weights.begin(), weights.end(), back_inserter(cumulative_weights));
//         }
//     }
//     return sample_indices;
// }

/*------------Python Bind--------------------------------------------------------------*/
PYBIND11_MODULE(sample_cores, m)
{
    m
    .def("neighbor_sample_from_nodes", 
        &neighbor_sample_from_nodes, 
        py::return_value_policy::reference)
    .def("neighbor_sample_from_nodes_with_time", 
        &neighbor_sample_from_nodes_with_time, 
        py::return_value_policy::reference)
    .def("neighbor_sample_from_nodes_with_before", 
        &neighbor_sample_from_nodes_with_before, 
        py::return_value_policy::reference)
    .def("get_neighbors", 
        &get_neighbors, 
        py::return_value_policy::reference)    
    .def("heads_unique", 
        &heads_unique, 
        py::return_value_policy::reference)
    .def("divide_nodes_to_part", 
        &divide_nodes_to_part, 
        py::return_value_policy::reference);

    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<vector<NodeIDType> &, vector<NodeIDType> &,
                      vector<NodeIDType> &>())
        .def("row", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.eid); })
        .def("nid", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.nid); })
        .def("e_ts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.e_ts); })
        .def("src_ts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.src_ts); })
        .def("sample_nodes", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.sample_nodes); })
        .def("sample_nodes_ts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.sample_nodes_ts); });

    py::class_<TemporalNeighborBlock>(m, "TemporalNeighborBlock")
        .def(py::init<vector<vector<NodeIDType>>&, 
                      vector<int64_t> &>())
        .def(py::pickle(
            [](const TemporalNeighborBlock& tnb) { return tnb.serialize(); },
            [](const std::string& s) { return TemporalNeighborBlock::deserialize(s); }
        ))
        .def("update_neighbors_with_time", 
            &TemporalNeighborBlock::update_neighbors_with_time)
        .def("update_edge_weight", 
            &TemporalNeighborBlock::update_edge_weight)
        // .def("get_node_neighbor",&TemporalNeighborBlock::get_node_neighbor)
        // .def("get_node_deg", &TemporalNeighborBlock::get_node_deg)
        .def_readonly("neighbors", &TemporalNeighborBlock::neighbors, py::return_value_policy::reference)
        .def_readonly("timestamp", &TemporalNeighborBlock::timestamp, py::return_value_policy::reference)
        .def_readonly("edge_weight", &TemporalNeighborBlock::edge_weight, py::return_value_policy::reference)
        .def_readonly("eid", &TemporalNeighborBlock::eid, py::return_value_policy::reference)
        .def_readonly("deg", &TemporalNeighborBlock::deg, py::return_value_policy::reference)
        .def_readonly("with_eid", &TemporalNeighborBlock::with_eid, py::return_value_policy::reference)
        .def_readonly("with_timestamp", &TemporalNeighborBlock::with_timestamp, py::return_value_policy::reference)
        .def_readonly("weighted", &TemporalNeighborBlock::weighted, py::return_value_policy::reference);
}