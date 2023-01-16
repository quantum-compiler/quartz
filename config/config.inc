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

if [ "$QUARTZ_USE_SIMULATOR" = "ON" ]; then
  SET_SIM="-DQUARTZ_USE_SIMULATOR=ON"
elif [ "$QUARTZ_USE_SIMULATOR" = "OFF" ]; then
  SET_SIM="-DQUARTZ_USE_SIMULATOR=OFF"
else
  SET_SIM="-DQUARTZ_USE_SIMULATOR=ON"
fi



SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../}
CMAKE_COMMAND="cmake ${SET_CUDA} ${SET_CUQUANTUM} ${SET_SIM} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
