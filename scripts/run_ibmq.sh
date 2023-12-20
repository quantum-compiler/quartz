#!/usr/bin/sh
cd ../build
make test_ibmq
mkdir -p ../circuit/nam-benchmarks/output_files/ibmq
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
./test_ibmq ../circuit/nam-benchmarks/adder_8.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/adder_8.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/barenco_tof_3.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/barenco_tof_3.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/barenco_tof_4.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/barenco_tof_4.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/barenco_tof_5.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/barenco_tof_5.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/barenco_tof_10.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/barenco_tof_10.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/csla_mux_3.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/csla_mux_3.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/csum_mux_9.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/csum_mux_9.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^4_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^4_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^5_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^5_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^6_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^6_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^7_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^7_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^8_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^8_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^9_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^9_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/gf2^10_mult.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/gf2^10_mult.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/mod5_4.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/mod5_4.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/mod_mult_55.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/mod_mult_55.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/mod_red_21.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/mod_red_21.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/qcla_adder_10.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/qcla_adder_10.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/qcla_com_7.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/qcla_com_7.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/qcla_mod_7.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/qcla_mod_7.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/rc_adder_6.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/rc_adder_6.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/tof_3.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/tof_3.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/tof_4.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/tof_4.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/tof_5.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/tof_5.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/tof_10.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/tof_10.qasm.output.ibmq $eqset_arg &
./test_ibmq ../circuit/nam-benchmarks/vbe_adder_3.qasm --output ../circuit/nam-benchmarks/output_files/ibmq/vbe_adder_3.qasm.output.ibmq $eqset_arg &
