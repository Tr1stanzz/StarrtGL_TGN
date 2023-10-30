import numpy as np
import torch
from multiprocessing import shared_memory

def _copy_to_share_memory(data):
    
    data_array=data.numpy() 
    shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
    data_share = np.ndarray(data_array.shape, dtype=data_array.dtype, buffer=shm.buf)
    np.copyto(data_share,data_array)
    name = shm.name
    _close_existing_shm(shm)
    #data_share.copy_(data_array) # Copy the original data into shared memory
    return name,data_array.shape,data_array.dtype

def _copy_to_shareable_list(data):
    if data is None:
        return None
    shm = shared_memory.ShareableList(data)
    name = shm.shm.name
    return name
def _get_from_shareable_list(name):
    if name is None:
        return None
    return shared_memory.ShareableList(name=name)

def _get_existing_share_memory(name):
    if name is None:
        return None
    return shared_memory.SharedMemory(name=name)

def _get_from_share_memory(existing_shm,data_shape,data_dtype):
    if existing_shm is None:
        return None
    data = np.ndarray(data_shape, dtype=data_dtype, buffer=existing_shm.buf)
    return torch.from_numpy(data)

def _close_existing_shm(existing_shm):
    if existing_shm is None:
        return 
    if(existing_shm != None):
        existing_shm.close()


def _unlink_existing_shm(existing_shm):
    if existing_shm is None:
        return 
    if(existing_shm != None):
        existing_shm.unlink()