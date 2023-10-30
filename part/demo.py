
from Utils import GraphData
from torch_geometric.data import Data
g = GraphData('./rpc_ps/part/metis_4/rank_0')
x= Data()
print(g)
'''
GraphData(
  partition_id=0
  data=Data(x=[679, 1433], edge_index=[2, 2908], y=[679], train_mask=[679], val_mask=[679], test_mask=[679]),
  global_info(num_nodes=2029, num_edges=10556, num_parts=4, edge_index=[2,10556])
)


'''

