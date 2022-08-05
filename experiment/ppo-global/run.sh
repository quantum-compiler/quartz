export OMP_NUM_THREADS=4

python ppo.py c=nam c.resume=false c.ddp_port=23373 \
    'c.input_graphs=[{ name: "barenco_tof_3", path: "../nam_circs/barenco_tof_3.qasm" }]' \
    c.mini_batch_size=600 c.k_epochs=10 \
    2>&1 | tee log/barenco_tof_3.log
