
from enum import Enum
import sys
import argparse
from os.path import abspath, join, dirname
import time

sys.path.insert(0, join(abspath(dirname(__file__))))
class SampleType(Enum):
    Whole = 0
    Inner = 1
    Outer =2

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--world_size', default=1, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='rank of the worker')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
parser.add_argument('--num_sampler', type=int, default=1, metavar='S',
                    help='number of samplers')
parser.add_argument('--queue_size', type=int, default=10000, metavar='S',
                    help='sampler queue size')
#parser = distparser.parser.add_subparsers().add_parser("train")#argparse.ArgumentParser(description='minibatch_gnn_models')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wiki or reddit)',
                    default='wiki')
parser.add_argument('--bs', type=int, default=600, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--fanout', type=int, default=10, help='Fanout of Sampler')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--recent', action='store_true',
                    help='take recent sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--use_inner_product', action='store_true',help='Whether to use inner product as score function')
args = parser.parse_args()
rpc_proxy=None
WORKER_RANK = args.rank
NUM_SAMPLER = args.num_sampler
WORLD_SIZE = args.world_size
QUEUE_SIZE = args.queue_size
MAX_QUEUE_SIZE = 5*args.queue_size
RPC_NAME = "rpcserver{}"
SAMPLE_TYPE = SampleType.Outer
def _get_worker_rank():
    return WORKER_RANK
def _get_num_sampler():
    return NUM_SAMPLER
def _get_world_size():
    return WORLD_SIZE
def _get_RPC_NAME():
    return RPC_NAME
def _get_queue_size():
    return QUEUE_SIZE
def _get_max_queue_size():
    return MAX_QUEUE_SIZE
def _get_rpc_name():
    return RPC_NAME