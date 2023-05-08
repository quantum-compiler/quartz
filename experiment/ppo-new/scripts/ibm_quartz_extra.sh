#!/bin/bash

set -ex

export OMP_NUM_THREADS=8

python scripts/ibm_bfs.py dj_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py ghz_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py graphstate_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py wstate_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py vqe_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qgan_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qaoa_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py ae_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qpeexact_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qpeinexact_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qft_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py qftentangled_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py portfoliovqe_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py realamprandom_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py twolocalrandom_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online & sleep 30
python scripts/ibm_bfs.py su2random_nativegates_ibm_tket_8 quartz_ibm_extra_outputs online

echo "Finished!"
