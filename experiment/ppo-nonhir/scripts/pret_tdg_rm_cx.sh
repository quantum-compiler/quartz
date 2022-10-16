#!/bin/bash

set -ex

date
pwd
ls -lh

mkdir -p logs

export OMP_NUM_THREADS=8

python ppo.py c=tdg_rm_mp c.gnn_type=qgnn c.mini_batch_size=4800 c.ddp_port=23333 'c.gpus=[0,1,2,3]' c.k_epochs=25 c.cost_type=cx_count 2>&1 | tee logs/tdg_cx_rm_mp_1004.log
