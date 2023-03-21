#!/bin/bash

set -ex

export OMP_NUM_THREADS=8

python scripts/nam2_rm_bfs.py adder_8 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py barenco_tof_3 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py barenco_tof_4 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py barenco_tof_5 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py barenco_tof_10 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py csla_mux_3 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py csum_mux_9 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^4_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^5_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^6_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^7_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^8_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^9_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^10_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^16_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py gf2^32_mult quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py grover_5 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py ham15-high quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py ham15-low quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py ham15-med quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py hwb6 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py mod_adder_1024 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py mod_mult_55 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py mod_red_21 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py mod5_4 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py qcla_adder_10 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py qcla_com_7 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py qcla_mod_7 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py rc_adder_6 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py tof_3 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py tof_4 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py tof_5 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py tof_10 quartz_nam2_rm_outputs online & sleep 30
python scripts/nam2_rm_bfs.py vbe_adder_3 quartz_nam2_rm_outputs online

echo "Finished!"
