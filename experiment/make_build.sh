#!/bin/bash

set -x	# print commands and their arguments as they are executed
set -e	# exit immediately if anything you're running returns a non-zero return code

cd ..
pwd

cd build
cmake -D CMAKE_INSTALL_PREFIX:PATH=~/usr_local ..
make install

cd ..
cd python

python setup.py build_ext --inplace install

