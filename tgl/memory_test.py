import argparse
import os
import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__)+'/startGNN_sample')))
from startGNN_sample.DistGraphLoader import DataSet,  partition_load
from startGNN_sample.Sample.temporal_neighbor_sampler import TemporalNeighborSampler
from startGNN_sample.Sample.base import NegativeSampling
#path1=os.path.abspath('.')  
import torch
from startGNN_sample.DistGraphLoader import DistGraphData
from startGNN_sample.DistGraphLoader import DistributedDataLoader
from startGNN_sample.DistGraphLoader import DistCustomPool 
import startGNN_sample.distparser as distparser
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
"""
test command 
python test.py --world_size 2 --rank 0 
--world_size', default=4, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='rank of the worker')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
parser.add_argument('--num_sampler', type=int, default=10, metavar='S',
                    help='number of samplers')
parser.add_argument('--queue_size', type=int, default=10, metavar='S',
                    help='sampler queue size')
"""
from tgl.modules import *
from tgl.sampler import *
from tgl.utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import dgl
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():   
    args = distparser.args

    DistCustomPool.init_distribution('127.0.0.1',9675,'127.0.0.1',10023,backend = "gloo")
    pdata = partition_load("./startGNN_sample/wiki", algo="metis")    
    graph = DistGraphData(pdata = pdata,edge_index= pdata.edge_index, full_edge = False)
    sampler = TemporalNeighborSampler(num_nodes=graph.num_nodes, num_layers=2, fanout=[10,10], graph_data=graph, workers=10,is_root_ts = True,graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,graph.data.train_mask).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,graph.data.train_mask)
    val_data = torch.masked_select(graph.edge_index,graph.data.val_mask).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,graph.data.val_mask)
    test_data = torch.masked_select(graph.edge_index,graph.data.test_mask).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,graph.data.test_mask) 
    train_data = DataSet(edges = train_data,ts =train_ts,labels = torch.ones(train_data.shape[1]))
    test_data = DataSet(edges = test_data,ts =test_ts,labels = torch.ones(test_data.shape[1]))
    val_data = DataSet(edges = val_data,ts = val_ts,labels = torch.ones(val_data.shape[1]))
    neg_sampler = NegativeSampling('triplet')
    trainloader = DistributedDataLoader('train',graph,train_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = 1)
    testloader = DistributedDataLoader('test',graph,test_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)
    valloader = DistributedDataLoader('val',graph,val_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)



    val_losses = list()
    def eval(mode='val'):
        neg_samples = 1
        model.eval()
        aps = list()
        aucs_mrrs = list()
        if mode == 'val':
            loader = valloader
        elif mode == 'test':
            loader = testloader
        elif mode == 'train':
            loader = trainloader
        with torch.no_grad():
            total_loss = 0
            for batchData in loader:
                mfgs,metadata = batch_data_prepare_input(batchData,sample_param['history'],cuda = use_cuda)
                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs,metadata)
                total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                aps.append(average_precision_score(y_true, y_pred))
                if neg_samples > 1:
                    aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
                else:
                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mode == 'val':
                val_losses.append(float(total_loss))
        ap = float(torch.tensor(aps).mean())
        if neg_samples > 1:
            auc_mrr = float(torch.cat(aucs_mrrs).mean())
        else:
            auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        return ap, auc_mrr

# set_seed(0)

    sample_param, memory_param, gnn_param, train_param = parse_config('tgl/config/TGN.yml')
    
    gnn_dim_node = 0 if graph.data.x is None else graph.data.x.shape[1]
    gnn_dim_edge = 0 if graph.data.edge_attr is None else graph.data.edge_attr.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
    use_cuda = False
    if use_cuda:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
    else:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first)
    model = DDP(model)
    
    mailbox = MailBox(memory_param, graph.partptr[-1], gnn_dim_edge) if memory_param['type'] != 'none' else None
    
    
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        #if node_feats is not None:
        #    node_feats = node_feats.cuda()
        #if edge_feats is not None:
        #    edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()

    sampler = None
#if not ('no_sample' in sample_param and sample_param['no_sample']):
#    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
#                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
#                              sample_param['strategy']=='recent', sample_param['prop_time'],
#                              sample_param['history'], float(sample_param['duration']))
#    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    neg_link_sampler = None#NegativeSampling()
    if not os.path.isdir('models'):
        os.mkdir('models')
    #if args.model_name == '':
    #    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
    #else:
    #    path_saver = 'models/{}.pkl'.format(args.model_name)
    best_ap = 0
    best_e = 0
    sample_time = time.time()
    for e in range(train_param['epoch']):
        train_aps = list()
        print('Epoch {:d}:'.format(e))
        time_prep = 0
        time_tot = 0
        total_loss = 0
        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
        last_time = time.time()
        for batchData in trainloader:
            
            sample_time += time.time() - last_time
        ##for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
            t_tot_s = time.time()
            t_prep_s = time.time()
            mfgs,metadata = batch_data_prepare_input(batchData,sample_param['history'],cuda = use_cuda)
            #if mailbox is not None:
            #    mailbox.prep_input_mails(mfgs[0])
            time_prep += time.time() - t_prep_s
            optimizer.zero_grad()
            
            pred_pos, pred_neg = model(mfgs,metadata)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            
            total_loss += float(loss) * train_param['batch_size']
            
            loss.backward()
            
            optimizer.step()
            t_prep_s = time.time()
            #if mailbox is not None:
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            train_aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
            #    eid = rows['Unnamed: 0'].values
            #    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            #    block = None
            #    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats,# block)
            #    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater#.last_updated_ts)
            time_prep += time.time() - t_prep_s
            time_tot += time.time() - t_tot_s
            last_time = time.time()
        train_ap = float(torch.tensor(train_aps).mean())    
        ap, auc = eval('val')
        if e > 2 and ap > best_ap:
            best_e = e
            best_ap = ap
        #    torch.save(model.state_dict(), path_saver)
        print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss,train_ap, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, sample_time, time_prep))    
#
    print('Loading model at epoch {}...'.format(best_e))
    #model.load_state_dict(torch.load(path_saver))
    model.eval()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
        eval('train')
        eval('val')
    ap, auc = eval('test')
    if args.eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))    

    DistCustomPool.close_distribution()
if __name__ == "__main__":
    main()
