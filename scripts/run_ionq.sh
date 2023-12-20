#!/bin/bash

set -ex

cd ../src/test

python test_ionq.py dj_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py ghz_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py graphstate_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py grover-noancilla_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py qft_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py qftentangled_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py qpeexact_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py qpeinexact_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py qwalk-noancilla_nativegates_ionq_qiskit_opt0_10_norm ../../ionq_output & sleep 60
python test_ionq.py shor_15_4_nativegates_ionq_qiskit_opt0_18_norm ../../ionq_output & sleep 60
python test_ionq.py shor_9_4_nativegates_ionq_qiskit_opt0_18_norm ../../ionq_output & sleep 60
