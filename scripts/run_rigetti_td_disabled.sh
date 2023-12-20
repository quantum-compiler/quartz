#!/usr/bin/sh
cd ../build
make test_rigetti_td_disabled
mkdir -p ../circuit/nam-benchmarks/output_files/rigetti
./test_rigetti_td_disabled ../circuit/nam-benchmarks/adder_8.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/adder_8.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/barenco_tof_3.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/barenco_tof_3.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/barenco_tof_4.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/barenco_tof_4.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/barenco_tof_5.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/barenco_tof_5.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/barenco_tof_10.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/barenco_tof_10.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/csla_mux_3.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/csla_mux_3.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/csum_mux_9.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/csum_mux_9.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^4_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^4_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^5_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^5_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^6_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^6_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^7_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^7_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^8_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^8_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^9_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^9_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/gf2^10_mult.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/gf2^10_mult.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/mod5_4.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/mod5_4.qasm.output.nrigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/mod_mult_55.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/mod_mult_55.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/mod_red_21.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/mod_red_21.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/qcla_adder_10.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/qcla_adder_10.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/qcla_com_7.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/qcla_com_7.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/qcla_mod_7.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/qcla_mod_7.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/rc_adder_6.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/rc_adder_6.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/tof_3.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/tof_3.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/tof_4.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/tof_4.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/tof_5.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/tof_5.qasm.output.rigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/tof_10.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/tof_10.qasm.output.nrigetti &
./test_rigetti_td_disabled ../circuit/nam-benchmarks/vbe_adder_3.qasm --output ../circuit/nam-benchmarks/output_files/rigetti/vbe_adder_3.qasm.output.rigetti &
