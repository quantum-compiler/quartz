cd ..
mkdir -p build
cd build
cmake ..
make gen_ecc_set
cd ..
./build/gen_ecc_set
