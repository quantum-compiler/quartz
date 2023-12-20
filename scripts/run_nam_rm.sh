#!/bin/bash

set -ex

cd ../src/test

python test_nam_rm.py adder_8 ../../nam_rm_output & sleep 60
python test_nam_rm.py barenco_tof_3 ../../nam_rm_output & sleep 60
python test_nam_rm.py barenco_tof_4 ../../nam_rm_output & sleep 60
python test_nam_rm.py barenco_tof_5 ../../nam_rm_output & sleep 60
python test_nam_rm.py barenco_tof_10 ../../nam_rm_output & sleep 60
python test_nam_rm.py csla_mux_3 ../../nam_rm_output & sleep 60
python test_nam_rm.py csum_mux_9 ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^4_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^5_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^6_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^7_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^8_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^9_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^10_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^16_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py gf2^32_mult ../../nam_rm_output & sleep 60
python test_nam_rm.py mod5_4 ../../nam_rm_output & sleep 60
python test_nam_rm.py mod_mult_55 ../../nam_rm_output & sleep 60
python test_nam_rm.py mod_red_21 ../../nam_rm_output & sleep 60
python test_nam_rm.py qcla_adder_10 ../../nam_rm_output & sleep 60
python test_nam_rm.py qcla_com_7 ../../nam_rm_output & sleep 60
python test_nam_rm.py qcla_mod_7 ../../nam_rm_output & sleep 60
python test_nam_rm.py rc_adder_6 ../../nam_rm_output & sleep 60
python test_nam_rm.py tof_3 ../../nam_rm_output & sleep 60
python test_nam_rm.py tof_4 ../../nam_rm_output & sleep 60
python test_nam_rm.py tof_5 ../../nam_rm_output & sleep 60
python test_nam_rm.py tof_10 ../../nam_rm_output & sleep 60
python test_nam_rm.py vbe_adder_3 ../../nam_rm_output
