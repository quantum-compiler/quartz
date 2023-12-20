#!/bin/bash

set -ex

cd ../src/test

python test_rigetti.py dj_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py ghz_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py graphstate_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py portfoliovqe_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py qftentangled_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py qft_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py qgan_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py realamprandom_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py su2random_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py twolocalrandom_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py vqe_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output & sleep 60
python test_rigetti.py wstate_nativegates_rigetti_qiskit_opt0_10_norm ../../rigetti_norm_output
