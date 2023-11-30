import os
import sys
from os.path import abspath, join, dirname
import torch
import yaml
import dgl
import time
import pandas as pd
import numpy as np

def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs



def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def batch_data_prepare_input_tgn(batch_data, cuda = True):
    sources_batch = batch_data.meta_data['src_id'].numpy()
    destinations_batch = batch_data.meta_data['dst_pos_id'].numpy()
    negatives_batch = batch_data.meta_data['dst_neg_id'].numpy()
    timestamps_batch = batch_data.roots.ts.numpy()
    pass

def batch_data_prepare_input(batch_data,hist,cuda = True,combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, type = 'identify'):
    mfgs = list()
    #e_index = torch.cat(batch_data.edge_index,1).view(-1)
    #e_id = torch.cat(batch_data.edge_id,1).view(-1)
    #e_ts = torch.cat(batch_data.edge_ts,1).view(-1)
    #src_ts = 
    #pre_torch_id = torch.cat((batch_data.meta_data['src_id'],batch_data.meta_data['dst_pos_id'],batch_data.meta_data['dst_neg_id'],e_index),dim = 0)
    #pre_torch_ts = torch.cat((batch_data.roots.ts.repeat(3),e_ts),dim = 0)
    #pre_eid = torch.cat((torch.zeros(len(batch_data.roots.ts)*3),e_id),dim = 0)
    #pre_src_ts = torch.cat((torch.zeros(len(batch_data.root.ts))))
    
    pre_torch  = torch.stack((torch.cat(batch_data.edge_index,1).view(-1),torch.cat(batch_data.edge_ts,1).reshape(-1)),dim = 0)
    pre_torch = torch.cat((torch.stack((torch.cat((batch_data.meta_data['src_id'],
                                                   batch_data.meta_data['dst_pos_id'],batch_data.meta_data['dst_neg_id']),dim = 0),
                     batch_data.roots.ts.repeat(3)),dim= 0 ),pre_torch),dim = -1)

    uniq,ind = pre_torch.unique(dim = 1, return_inverse = True)
    rt = len(batch_data.meta_data['src_id'])
    batch_data.meta_data['src_id_pos'] = ind[:rt]
    batch_data.meta_data['dst_pos_pos'] = ind[rt:2*rt]
    batch_data.meta_data['dst_neg_pos'] = ind[2*rt:3*rt]
    maxid = uniq.shape[-1]
    ind = ind[rt*3:].reshape(2,-1)
    lastindex = torch.zeros(uniq.shape[1]).long()
    id_l = 0
    l = 0
    last_src_data = None
    last_src_ts = None
    for i in range(len(batch_data.eids)):
        r = l + len(batch_data.eids[i])
        if i == 0:
            dst_nodes,pos = torch.cat((batch_data.meta_data['src_id_pos'],
                                                   batch_data.meta_data['dst_pos_pos'],batch_data.meta_data['dst_neg_pos']),dim = 0).unique(return_inverse=True)
            if cuda:
                batch_data.meta_data['src_id_pos'] = pos[:rt].cuda()
                batch_data.meta_data['dst_pos_pos'] = pos[rt:rt*2].cuda()
                batch_data.meta_data['dst_neg_pos'] = pos[rt*2:rt*3].cuda()
            else:
                batch_data.meta_data['src_id_pos'] = pos[:rt]
                batch_data.meta_data['dst_pos_pos'] = pos[rt:rt*2]
                batch_data.meta_data['dst_neg_pos'] = pos[rt*2:rt*3]
            dst_nodes_index = torch.arange(len(dst_nodes)).long()
            lastindex[dst_nodes.long()] = dst_nodes_index
            id_l += dst_nodes.shape[0]
            last_src_data = uniq[0,dst_nodes]
            last_src_ts = uniq[1,dst_nodes]
            src_nodes,src_nodes_index = ind[0,l:r].unique(return_inverse = True)
            #src_nodes_index += id_l
            b = dgl.create_block((src_nodes_index + id_l ,lastindex[ind[1,l:r]]),num_src_nodes = id_l + src_nodes.shape[0],num_dst_nodes = len(dst_nodes_index))
            lastindex[src_nodes.long()] = torch.arange(id_l,id_l + src_nodes.shape[0]).long()
            id_l += src_nodes.shape[0]
        else:
            src_nodes,src_nodes_index = ind[0,l:r].unique(return_inverse = True)
            #src_nodes_index += id_l
            b = dgl.create_block((src_nodes_index + id_l,lastindex[ind[1,l:r]]),num_src_nodes = id_l + src_nodes.shape[0],num_dst_nodes = id_l)
            lastindex[src_nodes.long()] = torch.arange(id_l,id_l + src_nodes.shape[0]).long()
            id_l += src_nodes.shape[0]
        l = r
        #b = dgl.create_block((ind[0,l:r],ind[1,l:r]),num_src_nodes= dim_in,num_dst_nodes= dim_out)
        last_src_data = torch.cat((last_src_data, uniq[0,src_nodes]),-1)
        last_src_ts = torch.cat((last_src_ts, uniq[1,src_nodes]),-1)
        b.srcdata['ID'] = last_src_data
        b.srcdata['dt'] = last_src_ts
        b.edata['dt'] = batch_data.edge_ts[i][1,:] - batch_data.edge_ts[i][0,:]
        b.edata['ID'] = batch_data.eids[i] 
        if cuda:
            b = b.to('cuda:0')
        
        j = 0
        if(batch_data.x is not None and i == len(batch_data.eids)-1 ):
            if pinned:
                idx = b.srcdata['ID'].cpu().long()
                torch.index_select(batch_data.x, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[j][:idx.shape[0]].cuda(non_blocking=True)
                j += 1
            else:
                srch = batch_data.x[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda() if cuda else srch
        j = 0
        if(batch_data.edge_attr is not None):
            if pinned:
                idx = b.edata['ID'].cpu().long()
                efeat_buffs[j][idx.shape[0]] = batch_data.edge_attr[idx]
                b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                j += 1
            else:
                srch = batch_data.edge_attr[b.edata['ID'].long()].float()
                b.edata['f'] = srch.cuda() if cuda else srch
                
        mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs,batch_data.meta_data
            

            
            
def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] =  [i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                eids.append(b.edata['ID'].long())
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

