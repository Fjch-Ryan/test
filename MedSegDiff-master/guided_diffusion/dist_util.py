"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    if not args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev   #默认为"0"

    backend = "gloo" if not th.cuda.is_available() else "nccl"    #nccl为GPU通信

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.1.1'#comm.bcast(hostname, root=0)   #主机地址
    os.environ["RANK"] = '0'#str(comm.rank)   #当前进程
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)  #总进程；这里只有一个进程

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)  #主机端口号
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.   MPI：多进程通信.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:  #“rb”表示以二进制·模式读取文件。在这种模式下，文件的内容被视为二进制数据，而不是文本数据。
            data = f.read()
    else:
        data = None
    #io.BytesIO(data)它接受一个字节字符串 data 作为参数，并将其作为初始数据加载到内存中。该函数返回一个 BytesIO 对象，该对象用于在内存中读取或写入二进制数据。
    #torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, **pickle_load_args)
    #输入参数：f ：一个类文件的对象(必须实现read()，:meth ' readline '，:meth ' tell '和:meth ' seek ')，或者一个字符串或os。包含文件名的类路径对象
    return th.load(io.BytesIO(data), **kwargs)     #th.load函数返回data数据的字典形式输出。


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)  #torch.distributed.broadcast是PyTorch分布式框架中的一个函数，它的作用是在分布式环境中将一个张量从指定的进程广播到所有其他进程。


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
