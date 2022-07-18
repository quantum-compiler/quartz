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

mkdir -p ftlog

export OMP_NUM_THREADS=4
# c.time_budget='3:30:00' \
python ppo.py c=nam_ft c.resume=true c.ckpt_path=pt_iter_115.pt c.ddp_port=23373 \
    'c.input_graphs=[{ name: "qcla_com_7", path: "../nam_circs/qcla_com_7.qasm" }]' \
    c.mini_batch_size=600 c.k_epochs=10 \
    2>&1 | tee ftlog/qcla_com_7.log

# python ppo.py c=nam_ft c.resume=true c.ckpt_path=cx_tdg_iter_80.pt c.ddp_port=23343 \
#        'c.input_graphs=[{ name: "gf2^9_mult", path: "../t_tdg_rm_circs/gf2^9_mult.qasm" }]' \
#        c.mini_batch_size=350 c.k_epochs=10 \
#        'c.gate_set=[ "h", "cx", "t", "tdg", "x" ]' c.ecc_file=../ecc_set/t_tdg.json.ecc c.cost_type=cx_count \
#        2>&1 | tee ftlog/cx_rm_gf2^9_mult.log

sleep 10


