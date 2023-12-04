import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from os.path import abspath, join, dirname
import sys
sys.path.insert(0, join(abspath(dirname(__file__))))
from partition.metis_part import partition_save

def compute_shift_time_statics(edge_ts, sources, destinations):
    timestamps = edge_ts
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


g_df = pd.read_csv('./data/wiki/ml_{}.csv'.format('wikipedia'))
e_feat = np.load('./data/wiki/ml_{}.npy'.format('wikipedia'))
n_feat = np.load('./data/wiki/ml_{}_node.npy'.format('wikipedia'))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_shift_time_statics(ts_l, src_l, dst_l)
print("Mean time shift src: {}, std: {}".format(mean_time_shift_src, std_time_shift_src))
print("Mean time shift dst: {}, std: {}".format(mean_time_shift_dst, std_time_shift_dst))

print(src_l.shape)
max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

valid_train_flag = (ts_l <= test_time)  
valid_val_flag = (ts_l <= test_time) 
assignment = np.random.randint(0, 10, len(valid_train_flag))
valid_train_flag *= (assignment >= 2)
valid_val_flag *= (assignment < 2)
valid_test_flag = ts_l > test_time

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

    # use the validation as test dataset
test_src_l = src_l[valid_val_flag]
test_dst_l = dst_l[valid_val_flag]
test_ts_l = ts_l[valid_val_flag]
test_e_idx_l = e_idx_l[valid_val_flag]
test_label_l = label_l[valid_val_flag]

data = Data()
data.edge_index = torch.cat((torch.from_numpy(np.array(src_l)).long()[np.newaxis, :],torch.from_numpy(np.array(dst_l)).long()[np.newaxis, :]),0)
# data.edge_index = torch.cat((data.edge_index,data.edge_index[torch.tensor([1,0])]),1)
data.edge_attr = torch.from_numpy(np.array(e_feat))[1:]
# data.edge_attr = torch.cat((data.edge_attr,data.edge_attr),0)
data.x = torch.from_numpy(np.array(n_feat))
data.train_mask =  (torch.from_numpy(np.array(valid_train_flag)))
# data.train_mask = torch.cat((data.train_mask,data.train_mask),0)
data.val_mask = torch.from_numpy(np.array(valid_val_flag))
# data.val_mask  = torch.cat((data.val_mask ,data.val_mask ),0)
data.test_mask = torch.from_numpy(np.array(valid_test_flag))
# data.test_mask = torch.cat((data.test_mask,data.test_mask),0)
data.edge_ts = torch.from_numpy(np.array(ts_l))
# data.edge_ts = torch.cat((data.edge_ts,data.edge_ts),0)
data.y = torch.from_numpy(np.array(label_l))
# data.y= torch.cat((data.y,data.y),0)
partition_save('./data/wiki', data, 2, 'metis')

