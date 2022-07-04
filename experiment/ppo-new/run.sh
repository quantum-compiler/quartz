#!/bin/bash

set -ex

date
pwd
ls -lh

if [ -f "~/colin/.proxy" ]; then
    echo "Find ~/colin/.proxy . Set proxy."
    source ~/colin/.proxy
elif [ -f "~/proxy" ]; then
    echo "Find ~/proxy . Set proxy."
    source ~/proxy
else 
    echo "Proxy not found."
fi

export OMP_NUM_THREADS=4

python ppo.py c=nam_ft c.resume=true c.ckpt_path=pt_iter_115.pt c.ddp_port=23373 \
    'c.input_graphs=[{ name: "gf2^5_mult", path: "../nam_circs/gf2^5_mult.qasm" }]' \
    c.time_budget='3:30:00' \
    c.mini_batch_size=600 c.k_epochs=10 \
    2>&1 | tee ftlog/gf2^5_mult.log

sleep 10


