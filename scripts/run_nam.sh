#!/usr/bin/sh
cd ../build
make test_nam
mkdir -p ../circuit/nam-benchmarks/output_files/nam
if [ $# -eq 0 ]
then
  eqset_arg=""
elif [ $# -eq 1 ]
then
  eqset_arg="--eqset $1"
else
  echo "Please pass only one argument for the ECC set file name."
  exit
fi
./test_nam ../circuit/nam-benchmarks/adder_8.qasm --output ../circuit/nam-benchmarks/output_files/nam/adder_8.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/barenco_tof_3.qasm --output ../circuit/nam-benchmarks/output_files/nam/barenco_tof_3.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/barenco_tof_4.qasm --output ../circuit/nam-benchmarks/output_files/nam/barenco_tof_4.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/barenco_tof_5.qasm --output ../circuit/nam-benchmarks/output_files/nam/barenco_tof_5.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/barenco_tof_10.qasm --output ../circuit/nam-benchmarks/output_files/nam/barenco_tof_10.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/csla_mux_3.qasm --output ../circuit/nam-benchmarks/output_files/nam/csla_mux_3.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/csum_mux_9.qasm --output ../circuit/nam-benchmarks/output_files/nam/csum_mux_9.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^4_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^4_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^5_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^5_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^6_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^6_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^7_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^7_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^8_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^8_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^9_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^9_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/gf2^10_mult.qasm --output ../circuit/nam-benchmarks/output_files/nam/gf2^10_mult.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/mod5_4.qasm --output ../circuit/nam-benchmarks/output_files/nam/mod5_4.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/mod_mult_55.qasm --output ../circuit/nam-benchmarks/output_files/nam/mod_mult_55.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/mod_red_21.qasm --output ../circuit/nam-benchmarks/output_files/nam/mod_red_21.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/qcla_adder_10.qasm --output ../circuit/nam-benchmarks/output_files/nam/qcla_adder_10.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/qcla_com_7.qasm --output ../circuit/nam-benchmarks/output_files/nam/qcla_com_7.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/qcla_mod_7.qasm --output ../circuit/nam-benchmarks/output_files/nam/qcla_mod_7.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/rc_adder_6.qasm --output ../circuit/nam-benchmarks/output_files/nam/rc_adder_6.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/tof_3.qasm --output ../circuit/nam-benchmarks/output_files/nam/tof_3.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/tof_4.qasm --output ../circuit/nam-benchmarks/output_files/nam/tof_4.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/tof_5.qasm --output ../circuit/nam-benchmarks/output_files/nam/tof_5.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/tof_10.qasm --output ../circuit/nam-benchmarks/output_files/nam/tof_10.qasm.output.nam $eqset_arg &
./test_nam ../circuit/nam-benchmarks/vbe_adder_3.qasm --output ../circuit/nam-benchmarks/output_files/nam/vbe_adder_3.qasm.output.nam $eqset_arg &
