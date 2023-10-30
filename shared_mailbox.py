
import sys
import torch
import dgl
from torch.distributed import scatter_object_list,scatter,gather,all_reduce,ReduceOp
import distparser
import torch_scatter
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef,rpc_async,remote,rpc_sync
mail_box = {}
#def update_mailbox(data_name,idx,memory):

class SharedMailBox():
    #
    def __init__(self,device = torch.device('cpu')):
        self.device = device
        self.rank = distparser._get_worker_rank()
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
    
    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')
        
    def build_map(self,store_type = 'local_node',local_num_nodes = 0,global_num_nodes = 0,edge_index = None,partptr = None):
        if store_type == 'undirect':
            nids = edge_index.view(-1)
            
        elif store_type == 'dst':
            self.nids = edge_index.view(-1).unique().to(self.device)
            self.num_nodes = len(nids)
            self.dict = torch.zeros(global_num_nodes,device = self.device,dtype=torch.int32)
            self.dict[nids] = torch.range(0,len(nids),device = self.device)
            self.mask = torch.BoolTensor(self.num_nodes,device = self.device)
        #处理rpc调用远程memory的情况
        elif store_type == 'local_node':
            self.nids = torch.arange(local_num_nodes,device = self.device)
            self.num_nodes = len(self.nids)
            num_part = distparser._get_world_size()
            self.partptr = partptr.to(self.device)
            
            
    def create_empty_memory(self,memory_param,dim_edge_feat,_node_memory = None,_node_memory_ts = None,_mailbox = None,_mailbox_ts = None,_next_mail_pos = None, _update_mail_pos = None,is_time_block = False):
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.memory_param = memory_param
        self.memory_size =  memory_param['dim_out']
        self.mail_size = [memory_param['mailbox_size'],2 * memory_param['dim_out'] + dim_edge_feat]
        self.node_memory = torch.zeros((self.num_nodes, memory_param['dim_out']), dtype=torch.float32,device =self.device) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(self.num_nodes, dtype=torch.float32,device = self.device) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros(self.num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat,device = self.device, dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((self.num_nodes, memory_param['mailbox_size']), dtype=torch.float32,device  = self.device) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((self.num_nodes), dtype=torch.long,device = self.device) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
    #远程拉取方案
    def update_local_memory(self,memory_info_list, reduce = 'max'):
        rank = self.rank
        nid = memory_info_list[:,0].reshape(-1).long()-self.partptr[rank]
        memory = memory_info_list[:,1:1+self.memory_size].to(self.node_memory.dtype)
        memory_ts = memory_info_list[:,1+self.memory_size].reshape(-1).to(self.node_memory_ts.dtype)
        #mail = memory_info_list[:,2+self.memory_size:2+self.memory_size+self.mail_size[1]]
        #mail_ts = memory_info_list[:,2+self.memory_size+self.mail_size[1]].reshape(-1)
        mail = memory_info_list[:,2+self.memory_size:2+self.memory_size+self.mail_size[1]].to(self.mailbox.dtype)
        mail_ts = memory_info_list[:,2+self.memory_size+self.mail_size[1]].reshape(-1).to(self.mailbox_ts.dtype)
        unq,inv = torch.unique(nid,return_inverse = True)
        #聚合方式
        if self.memory_param['mail_combine'] == 'last':
            max_mail_ts,mail_idx = torch_scatter.scatter_max(mail_ts,inv,0)
            self.mailbox_ts[unq.long(), self.next_mail_pos[unq]] = max_mail_ts
            self.mailbox[unq.long(), self.next_mail_pos[unq]] = mail[mail_idx]
            if self.memory_param['mailbox_size'] > 1:
                self.next_mail_pos[unq] = torch.remainder(self.next_mail_pos[unq] + 1, self.memory_param['mailbox_size'])
        if reduce == 'max':
            max_ts,idx = torch_scatter.scatter_max(memory_ts,inv,0)
            self.node_memory_ts[unq] = max_ts
            self.node_memory[unq] = memory[idx]
        elif reduce =='mean':
            self.node_memory_ts[unq]= torch_scatter.scatter_mean(memory_ts,inv[nid],0)
            self.node_memory_ts[unq] = torch_scatter.scatter_mean(memory,inv[nid],0)
        else:
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = memory_ts
    #不设置聚合策略，远程更新直接异步，随机更新,注意标记策略
    def update_memory_by_scatter(self,nid,memory,ts,mail = None,mail_ts = None,policy = 'all',reduce = 'max'):
        num_part = distparser._get_world_size()
        local_part = distparser._get_worker_rank()
        if policy == 'all':
            send_list = []
            rece_list = []
            rece_len_list = [torch.zeros(1,dtype=torch.long,device=self.device) for i in range(num_part)]
            for i in range(num_part):
                mask = ((nid<self.partptr[i+1]) & (nid>=self.partptr[i]))
                send_list.append(torch.cat([nid[mask].reshape(-1,1),memory[mask],ts[mask].reshape(-1,1),mail[mask],mail_ts[mask].reshape(-1,1)],1))
            for i in range(num_part):
                if i == self.rank:
                    gather(torch.tensor([send_list[i].shape[0]],device=self.device,dtype=torch.long),rece_len_list,i)
                else:
                    gather(torch.tensor([send_list[i].shape[0]],device=self.device,dtype=torch.long),None,i)
           
            len_tensor = torch.cat(rece_len_list).max().reshape(1)
            all_reduce(len_tensor, op=ReduceOp.MAX)
            maxlen = len_tensor.item()
            #for i in range(num_part):
            #    maxlen = max(maxlen,rece_len_list[i].item())
            rece_list = [torch.zeros(maxlen,send_list[i].shape[1],dtype = send_list[i].dtype,device=self.device) for i in range(num_part)]
            for i in range(num_part):    
                send = rece_list[0].clone()
                send[:send_list[i].shape[0],:]=send_list[i]
                if i == self.rank:
                    gather(send,rece_list,i)
                else:
                    gather(send,None,i)
            for i in range(num_part):
                rece_list[i] = rece_list[i][:rece_len_list[i]]
            update_info = torch.cat(rece_list,0)
            self.update_local_memory(update_info)
        elif policy == 'local':
            mask = nid<self.partptr[local_part+1] and nid>=self.partptr[local_part]
            self.update_local_memory(torch.cat([nid[mask],memory[mask],ts[mask].reshape(-1,1),mail[mask],mail_ts[mask].reshape(-1,1)],1))

    def prep_input_mails(self,nids,mfg):
        memory,memory_ts,mail,mail_ts = self.get_memory_by_scatter(nids)
        for i, b in enumerate(mfg):
            idx = b.srcdata['ID'].long()
            b.srcdata['mem'] = memory[idx]
            b.srcdata['mem_ts']  = memory_ts[idx]
            b.srcdata['mem_input'] = mail[idx].reshape(b.srcdata['ID'].shape[0], -1)
            b.srcdata['mail_ts'] = mail_ts[idx]
            
    def set_local_memory(self,nids,memory,memory_ts,mail,mail_ts):
        if nids.shape[0] == 0:
            return
        if(nids[0] >= self.partptr[self.rank]):
            nids = nids-self.partptr[self.rank]
        self.node_memory[nids] = memory 
        self.node_memory_ts[nids] = memory_ts
        self.mailbox_ts[nids.long(), self.next_mail_pos[nids]] = mail_ts
        self.mailbox[nids.long(), self.next_mail_pos[nids]] = mail
        if self.memory_param['mailbox_size'] > 1:
            self.next_mail_pos[nids.long()] = torch.remainder(self.next_mail_pos[nids.long()] + 1, self.memory_param['mailbox_size'])
    def get_local_memory(self,nids):
        if(nids.shape[0]>0 and nids[0] >= self.partptr[self.rank]):
            nids = nids-self.partptr[self.rank]
        return self.node_memory[nids],self.node_memory_ts[nids],self.mailbox[nids],self.mailbox_ts[nids]
    
    def get_memory_by_scatter(self,nids):
        num_part = distparser._get_world_size()
        rank = self.rank 
        nid_list = []
        len_list = []
        maxlen = 0
        for i in range(num_part):
            #print((nids<self.partptr[i+1]),(nids>=self.partptr[i]))
            mask =  ((nids<self.partptr[i+1]) & (nids>=self.partptr[i]))
            nid_i = nids[mask]
            nid_list.append(nid_i)
            len_list.append(torch.tensor(nid_i.shape[0],device=self.device).reshape(1))
            maxlen = max(maxlen,len_list[i].item())
        send_out_len_list = [torch.tensor([0],device =  self.device) for _ in range(num_part)]          
        for i in range(num_part):
            if i == self.rank:
                gather(len_list[i],send_out_len_list,i)
            else:
                gather(len_list[i],None,i)
        max_sent = torch.cat(send_out_len_list).max().reshape(1)
        all_reduce(max_sent, op=ReduceOp.MAX)
        maxlen = max_sent.item()
        #for i in range(num_part):
        #    max_sent = max(max_sent,send_out_len_list[i].item())
        
        #send_out_nid_list = [torch.zeros(100,dtype = torch.long,device =self.device) for i in range(num_part)]  
        send_out_nid_list = [torch.zeros(max_sent,dtype = torch.long,device =self.device) for i in range(num_part)]   
        #print(max_sent,send_out_len_list[0],send_out_len_list[1],nid_list[0].shape,nid_list[1].shape)
        for i in range(num_part):
            send = send_out_nid_list[0].clone()
            send[:nid_list[i].shape[0]] = nid_list[i]
            if i == self.rank:
                gather(send,send_out_nid_list,i)
            else:
                gather(send,None,i)
        torch.distributed.barrier()
        for i in range(num_part):
            send_out_nid_list[i] = send_out_nid_list[i][0:send_out_len_list[i]]

        outer_nid = torch.cat(send_out_nid_list,0)-self.partptr[rank]
        memory_info = torch.cat([self.node_memory[outer_nid],self.node_memory_ts[outer_nid].view(-1,1),self.mailbox[outer_nid].view(-1,self.mail_size[0]*self.mail_size[1]),self.mailbox_ts[outer_nid].view(-1,1)],1)
        #max_sent = 600
        #print(max_sent)
        rece_len = [torch.zeros(1,device = self.device,dtype = torch.long) for i in range(num_part)]
        torch.distributed.all_gather(rece_len,torch.tensor([maxlen],device = self.device,dtype =torch.long))
        rece_in_memory = [torch.zeros(maxlen,memory_info.shape[1],dtype=self.node_memory.dtype,device = self.device) for i in range(num_part)]
        for i in range(1,num_part):    
            send_out_len_list[i] = send_out_len_list[i-1]+send_out_len_list[i]
        for i in range(num_part):
            send = torch.zeros(rece_len[i],memory_info.shape[1],dtype=self.node_memory.dtype,device = self.device)
            if i == 0:
                send[:send_out_len_list[i],:] = memory_info[:send_out_len_list[i],:]
                if i == self.rank:    
                    gather(send,rece_in_memory,i)
                else:
                    gather(send,None,i)
            else:
                send[:send_out_len_list[i]-send_out_len_list[i-1],:] = memory_info[send_out_len_list[i-1]:send_out_len_list[i],:]
                if i ==self.rank:
                    gather(send,rece_in_memory,i)
                else:
                    gather(send,None,i)
        recv_in_final = [rece_in_memory[self.rank][0:len_list[self.rank].item(),:]]
        for i in range(num_part):
            if i == self.rank:
                continue
            else:
                recv_in_final.append(rece_in_memory[i][0:len_list[i].item(),:])
            #rece_in_memory[i] = rece_in_memory[i][0:len_list[i].item(),:]
        
        #rece_in_memory = torch.cat(rece_in_memory,0)
        rece_in_memory = torch.cat(recv_in_final,0)
        memory = rece_in_memory[:,:self.memory_size]
        memory_ts = rece_in_memory[:,self.memory_size].reshape(-1)
        mail = rece_in_memory[:,1+self.memory_size:1+self.memory_size+self.mail_size[1]*self.mail_size[0]].view(-1,self.mail_size[0],self.mail_size[1])
        mail_ts = rece_in_memory[:,1+self.memory_size+self.mail_size[1]*self.mail_size[0]].reshape(-1)
        return memory,memory_ts,mail,mail_ts
    
    def get_memory_to_update(self,block_ptr,src,dst,ts,edge_feats,memory,memory_ts,policy = 'all',reduce='max'):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        # TGN/JODIE
        src = src.to(self.device)
        dst = dst.to(self.device)
        mem_src = memory[src]
        mem_dst = memory[dst]
        if edge_feats is not None:
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
        else:
            src_mail = torch.cat([mem_src, mem_dst], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src], dim=1)
        mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)
        #if mail_ts.dtype == torch.float64:
        #    import pdb; pdb.set_trace()
        # find unique nid to update mailbox
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mail = mail[perm]
        mail_ts = mail_ts[perm]
        memory = memory[nid]
        memory_ts = memory_ts[nid]
        nid = block_ptr[nid]
        self.update_memory_by_scatter(nid,memory,memory_ts,mail,mail_ts,policy = policy,reduce = reduce)

local_rref = []      
def get_local_rref():
    return local_rref     
class SharedRPCMemoryManager(SharedMailBox):
    def __init__(self,device = torch.device('cpu')):
        super(SharedRPCMemoryManager,self).__init__(device)
        self.remote_rref = []
        self.num_part = distparser._get_world_size()
        #rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
        #rpc_backend_options.init_method = "tcp://{}:{}".format(master,port)
        #for i in range(self.num_part):
        #    if i == self.rank:
        #            continue
        #    if(device == torch.device('cpu')):
        #        rpc_backend_options.set_device_map(f"{i}",{self.device:self.device})
        #    else:
        #        rpc_backend_options.set_device_map(f"{i}",{self.rank:i})
#
        #rpc.init_rpc(f"mem-{self.rank}",rank = self.rank,world_size = self.num_part,rpc_backend_options=rpc_backend_options)
        rref = RRef(self)
        global local_rref
        local_rref = rref
        torch.distributed.barrier()
        remote_rref = []
        for i in range(self.num_part):
            if i == self.rank:
                self.remote_rref.append(rref)
                continue
            rev = rpc_sync(distparser._get_rpc_name().format(i)+"-{}".format(i*(distparser._get_num_sampler()+1)+1),get_local_rref,args=None)
            self.remote_rref.append(rev)
        
        if self.device != torch.device('cpu'):
            torch.cuda.current_stream("cuda").synchronize()
    
    def get_memory_by_scatter(self, nids):
        futs = []
        
        for i in range(self.num_part):
            #print((nids<self.partptr[i+1]),(nids>=self.partptr[i]))
            mask =  ((nids<self.partptr[i+1]) & (nids>=self.partptr[i]))
            nid_i = nids[mask]
            if i == self.rank:
                local_m = self.get_local_memory(nid_i)
                futs.append([])
            else:
                futs.append(self.remote_rref[i].rpc_async().get_local_memory(nid_i))
        memory = [local_m[0]]
        memory_ts =[local_m[1]]
        mailbox = [local_m[2]]
        mailbox_ts =[local_m[3]]
        for i in range(self.num_part):
            if i == self.rank:
                continue
            else:
                futs[i].wait()
                mem = futs[i].value()
                memory.append(mem[0])
                memory_ts.append(mem[1])
                mailbox.append(mem[2])
                mailbox_ts.append(mem[3])
        return torch.cat(memory,0),torch.cat(memory_ts,0),torch.cat(mailbox,0),torch.cat(mailbox_ts,0)
    def prep_input_mails(self,nids,mfg):
        memory,memory_ts,mail,mail_ts = self.get_memory_by_scatter(nids)
        for i, b in enumerate(mfg):
            idx = b.srcdata['ID'].long()
            b.srcdata['mem'] = memory[idx]
            b.srcdata['mem_ts']  = memory_ts[idx]
            b.srcdata['mem_input'] = mail[idx].reshape(b.srcdata['ID'].shape[0], -1)
            b.srcdata['mail_ts'] = mail_ts[idx]
    def get_memory_to_update(self,block_ptr,src,dst,ts,edge_feats,memory,memory_ts,policy = 'all',reduce='max'):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        # TGN/JODIE
        src = src.to(self.device)
        dst = dst.to(self.device)
        mem_src = memory[src]
        mem_dst = memory[dst]
        if edge_feats is not None:
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
        else:
            src_mail = torch.cat([mem_src, mem_dst], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src], dim=1)
        mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)
        #if mail_ts.dtype == torch.float64:
        #    import pdb; pdb.set_trace()
        # find unique nid to update mailbox
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mail = mail[perm]
        mail_ts = mail_ts[perm]
        memory = memory[nid]
        memory_ts = memory_ts[nid].to(self.node_memory_ts.dtype)
        nids = block_ptr[nid]
        for i in range(self.num_part):
            mask =  ((nids<self.partptr[i+1]) & (nids>=self.partptr[i]))
            nid_i = nids[mask]
            mail_i  = mail[mask]
            mail_ts_i = mail_ts[mask]
            memory_i = memory[mask]
            memory_ts_i = memory_ts[mask]
            if i == self.rank:
                self.set_local_memory(nid_i,memory_i,memory_ts_i,mail_i,mail_ts_i)
            else:
                self.remote_rref[i].rpc_async().set_local_memory(nid_i,memory_i,memory_ts_i,mail_i,mail_ts_i)


