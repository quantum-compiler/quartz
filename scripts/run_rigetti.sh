#!/usr/bin/sh
cd ../build
make test_rigetti
if [ $# -eq 0 ]
then
  eqset_arg=""
  output_dir="rigetti"
elif [ $# -eq 1 ]
then
  eqset_arg="--eqset $1"
  output_dir="rigetti"
elif [ $# -eq 2 ]
then
  eqset_arg="--eqset $1"
  output_dir=$2
else
  echo "Please pass only one argument for the ECC set file name (and possibly another argument for output directory)."
  exit
fi
mkdir -p ../circuit/nam-benchmarks/output_files/$2
./test_rigetti ../circuit/nam-benchmarks/adder_8.qasm --output ../circuit/nam-benchmarks/output_files/$2/adder_8.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/barenco_tof_3.qasm --output ../circuit/nam-benchmarks/output_files/$2/barenco_tof_3.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/barenco_tof_4.qasm --output ../circuit/nam-benchmarks/output_files/$2/barenco_tof_4.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/barenco_tof_5.qasm --output ../circuit/nam-benchmarks/output_files/$2/barenco_tof_5.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/barenco_tof_10.qasm --output ../circuit/nam-benchmarks/output_files/$2/barenco_tof_10.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/csla_mux_3.qasm --output ../circuit/nam-benchmarks/output_files/$2/csla_mux_3.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/csum_mux_9.qasm --output ../circuit/nam-benchmarks/output_files/$2/csum_mux_9.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^4_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^4_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^5_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^5_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^6_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^6_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^7_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^7_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^8_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^8_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^9_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^9_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/gf2^10_mult.qasm --output ../circuit/nam-benchmarks/output_files/$2/gf2^10_mult.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/mod5_4.qasm --output ../circuit/nam-benchmarks/output_files/$2/mod5_4.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/mod_mult_55.qasm --output ../circuit/nam-benchmarks/output_files/$2/mod_mult_55.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/mod_red_21.qasm --output ../circuit/nam-benchmarks/output_files/$2/mod_red_21.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/qcla_adder_10.qasm --output ../circuit/nam-benchmarks/output_files/$2/qcla_adder_10.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/qcla_com_7.qasm --output ../circuit/nam-benchmarks/output_files/$2/qcla_com_7.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/qcla_mod_7.qasm --output ../circuit/nam-benchmarks/output_files/$2/qcla_mod_7.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/rc_adder_6.qasm --output ../circuit/nam-benchmarks/output_files/$2/rc_adder_6.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/tof_3.qasm --output ../circuit/nam-benchmarks/output_files/$2/tof_3.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/tof_4.qasm --output ../circuit/nam-benchmarks/output_files/$2/tof_4.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/tof_5.qasm --output ../circuit/nam-benchmarks/output_files/$2/tof_5.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/tof_10.qasm --output ../circuit/nam-benchmarks/output_files/$2/tof_10.qasm.output.rigetti $eqset_arg &
./test_rigetti ../circuit/nam-benchmarks/vbe_adder_3.qasm --output ../circuit/nam-benchmarks/output_files/$2/vbe_adder_3.qasm.output.rigetti $eqset_arg &
