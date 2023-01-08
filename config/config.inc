#!/bin/bash

function run_cmake() {

# set CUDA dir
if [ -n "$CUDA_DIR" ]; then
  SET_CUDA="-DCUDA_PATH=${CUDA_DIR}"
fi

# set custatevec dir
if [ -n "$CUQUANTUM_DIR" ]; then
  SET_CUQUANTUM="-DCUQUANTUM_ROOT=${CUQUANTUM_DIR}"
fi



SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../}
CMAKE_COMMAND="cmake ${SET_CUDA} ${SET_CUQUANTUM} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
