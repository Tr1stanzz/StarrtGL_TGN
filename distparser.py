
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
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
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