cd ..
mkdir -p build
cd build
cmake ..
make test_pruning
cd ..
./build/test_pruning
