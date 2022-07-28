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

from icecream import ic

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
        self.bn = nn.SyncBatchNorm(10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

        self.net3 = nn.Linear(10, 10)

    def forward(self, x, part=0):
        if part == 0:
            return self.net2(self.bn(self.relu(self.net1(x))))
        else:
            return self.net3(x)
    
    def forward2(self, x):
        return self.net1(x)
    
    def add_mod(self):
        self.net4 = nn.Linear(10, 10)
        return self


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU
    device = torch.device(f'cuda:{rank}')
    model = ToyModel().add_mod().to(device)
    # print(model.state_dict())
    torch.cuda.set_device(device)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[device])
    # print(f'sleeping...') # 1231+1097
    # time.sleep(20)
    local_model = ddp_model.module

    loss_fn = nn.MSELoss()
    # if rank == 0:
    #     print(list(ddp_model.parameters()))
    optimizer = optim.Adam([
        {
            'params': ddp_model.module.net1.parameters(),
            'lr': 1e-3,
        },
        {
            'params': ddp_model.module.net2.parameters(),
            'lr': 1e-4,
        }
    ])
    
    if rank == 0:
        ic(ddp_model.state_dict().values())
    return
    # state_dict = ddp_model.state_dict()
    # # assert state_dict is ddp_model.state_dict()
    # for k, v in state_dict.items():
    #     state_dict[k] = v + 100
    # if rank == 0:
    #     ic(ddp_model.state_dict())
    #     ic(state_dict)
    #     torch.save(state_dict, 'test_state.pt')
    # dist.barrier()
    # loaded_dict = torch.load('test_state.pt')
    # ddp_model.load_state_dict(loaded_dict)
    # if rank == 1:
    #     ic(ddp_model.state_dict())
    #     ic(model.state_dict())

    state_dict = model.state_dict()
    # assert state_dict is ddp_model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v + 100
    if rank == 0:
        ic(ddp_model.state_dict())
        ic(state_dict)
        torch.save(state_dict, 'test_state.pt')
    dist.barrier()
    loaded_dict = torch.load('test_state.pt')
    model.load_state_dict(loaded_dict)
    if rank == 1:
        ic(ddp_model.state_dict())

    assert model is ddp_model.module
    return

    # optimizer = optim.Adam(ddp_model.parameters())
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    x = torch.randn(20, 10) + rank * 100
    x = x.to(device)

    for i in range(1):
        optimizer.zero_grad()

        x = torch.randn(20, 10) #  + rank * 100
        x = x.to(device)
        labels = x + rank * 10000

        # outputs = ddp_model.module.forward2(x)
        ddp_model.eval()
        with torch.no_grad():
            y = ddp_model(torch.randn(20, 10), part=1)
        # ddp_model.train()
        outputs = ddp_model(x) +  ddp_model(torch.randn(20, 10), part=1)

        loss_fn(outputs, labels).backward()
        optimizer.step()
        print(f'rank {rank} finished {i}')
    
    print(f'rank {rank}:  {ddp_model.state_dict()}')
    print(f'rank {rank}:  local: {local_model.state_dict()}')
    
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(demo_basic, 4)
