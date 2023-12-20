#!/bin/bash

set -ex

cd ../src/test

python test_nam_worm.py adder_8 ../../nam_worm_output & sleep 60
python test_nam_worm.py barenco_tof_3 ../../nam_worm_output & sleep 60
python test_nam_worm.py barenco_tof_4 ../../nam_worm_output & sleep 60
python test_nam_worm.py barenco_tof_5 ../../nam_worm_output & sleep 60
python test_nam_worm.py barenco_tof_10 ../../nam_worm_output & sleep 60
python test_nam_worm.py csla_mux_3 ../../nam_worm_output & sleep 60
python test_nam_worm.py csum_mux_9 ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^4_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^5_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^6_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^7_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^8_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^9_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^10_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^16_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py gf2^32_mult ../../nam_worm_output & sleep 60
python test_nam_worm.py mod5_4 ../../nam_worm_output & sleep 60
python test_nam_worm.py mod_mult_55 ../../nam_worm_output & sleep 60
python test_nam_worm.py mod_red_21 ../../nam_worm_output & sleep 60
python test_nam_worm.py qcla_adder_10 ../../nam_worm_output & sleep 60
python test_nam_worm.py qcla_com_7 ../../nam_worm_output & sleep 60
python test_nam_worm.py qcla_mod_7 ../../nam_worm_output & sleep 60
python test_nam_worm.py rc_adder_6 ../../nam_worm_output & sleep 60
python test_nam_worm.py tof_3 ../../nam_worm_output & sleep 60
python test_nam_worm.py tof_4 ../../nam_worm_output & sleep 60
python test_nam_worm.py tof_5 ../../nam_worm_output & sleep 60
python test_nam_worm.py tof_10 ../../nam_worm_output & sleep 60
python test_nam_worm.py vbe_adder_3 ../../nam_worm_output
