import torch
import torch.nn as nn
import torch.multiprocessing as mp
from functools import partial


def train(task, net):
    print(f'task {task} : net')


def main():
    mp.set_start_method('spawn')
    net = nn.Linear(3, 5).cuda(0)
    net = net.share_memory()
    p_train = partial(train, net=net)
    
    with mp.Pool(
        processes=4,
        maxtasksperchild=1,
    ) as pool:
        pool.map(p_train, list(range(16)), chunksize=1)
    

if __name__ == '__main__':
    main()
