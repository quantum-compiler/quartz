import os
import time
from pickletools import optimize
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    
    def forward2(self, x):
        return self.net1(x)


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU
    device = torch.device(f'cuda:{rank+2}')
    model = ToyModel().to(device)
    # print(model.state_dict())
    torch.cuda.set_device(device)
    ddp_model = DDP(model, device_ids=[device])
    print(f'sleeping...') # 1231+1097
    time.sleep(20)

    loss_fn = nn.MSELoss()
    if rank == 0:
        print(list(ddp_model.parameters()))
    # optimizer = optim.Adam([
    #     {
    #         'params': ddp_model.module.net1.parameters(),
    #         'lr': 1e-3,
    #     },
    #     {
    #         'params': ddp_model.module.net2.parameters(),
    #         'lr': 1e-4,
    #     }
    # ])
    optimizer = optim.Adam(ddp_model.parameters())
    
    optimizer.zero_grad()
    x = torch.randn(20, 10) + rank * 100
    x = x.to(device)
    outputs = ddp_model.module.forward2(x)
    labels = torch.randn(20, 10).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    print(ddp_model.state_dict())
    
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(demo_basic, 2)
