import os
import pickle
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def obj_to_tensor(obj):
    s = pickle.dumps(obj)


def init_process(rank, tot_processes):
    print(f'here is rank {rank}', flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{52524}',
        rank=rank, world_size=tot_processes,
    )
    
    info_list = [[None, None] for _ in range(tot_processes)]
    info_list[rank] = [
        {f'{rank}': rank * i}
        for i in [3,5]
    ]
    
    for r in range(tot_processes):
        dist.broadcast_object_list(
            info_list[r], r
        )
    
    print(info_list, flush=True)
    
def init_process2(rank, tot_processes):
    print(f'here is rank {rank}', flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{52524}',
        rank=rank, world_size=tot_processes,
    )
    device = torch.device(f'cuda:{rank}')
    x = torch.zeros(4, 100).to(device)
    x[rank] += 1
    
    dist.barrier()
    
    for r in range(tot_processes):
        dist.broadcast_object_list(
            x[r], r, device=device
        )
    
    print(x, flush=True)


def main() -> None:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1000000, rlimit[1]))
    
    mp.set_start_method('spawn')
    tot_processes = 4
    print(f'spawning {tot_processes} processes...')
    mp.spawn(
        fn=init_process2,
        args=(tot_processes, ),
        nprocs=tot_processes,
        join=True,
    )

if __name__ == '__main__':
    main()
