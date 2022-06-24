import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

def work_func(x):
    return x * 100


def init_process(rank, tot_processes):
    print(f'here is rank {rank}', flush=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{52529}'
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        # init_method=f'tcp://localhost:52524',
        num_worker_threads=32,
    )
    rpc.init_rpc(
        name=f'test_{rank}', rank=rank, world_size=tot_processes,
        rpc_backend_options=rpc_backend_options,
    )
    print(f'rank {rank} init successfully', flush=True)

    if rank == 0:
        ret_list = [
            rpc.rpc_sync(f'test_{i}', work_func, args=(i,))
            for i in range(1, tot_processes)
        ]
        print(ret_list)

    rpc.shutdown()


def main() -> None:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1000000, rlimit[1]))
    
    mp.set_start_method('spawn')
    tot_processes = 19
    print(f'spawning {tot_processes} processes...')
    mp.spawn(
        fn=init_process,
        args=(tot_processes, ),
        nprocs=tot_processes,
        join=True,
    )

if __name__ == '__main__':
    main()
