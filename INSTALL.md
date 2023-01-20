# Quartz Installation

Quartz can be built from source code using the following instructions.

## Install from Source

### Prerequisites

* `apt install build-essential`
* CMAKE 3.16 or higher: `apt install cmake`
* Cython 3.0 or higher (We will install it with Python later by `conda`)

### Build Quartz Runtime

* To get started, clone the Quartz source code from GitHub.
```shell
git clone --recursive https://github.com/quantum-compiler/quartz.git
cd quartz
```

* Set up the environment (including Python, Cython, OpenMP) by conda.

```shell
conda env create --name quartz --file env.yml
conda activate quartz
```

* Build the Quartz runtime library. The configuration of the Quartz runtime can be modified by `config.cmake`.

```shell
mkdir build;
cd build;
cmake .. # see notes below
make install
```

Note that line 3 in the example will have the runtime library and include files installed into the default path `/usr/local/`. To modify the install path, you can set the path explicitly in line 3, for example:

```shell
cmake -D CMAKE_INSTALL_PREFIX:PATH=~/opt/ ..
```


* Install the Quartz python package.

```shell
cd ../python
python setup.py build_ext --inplace install
```

Note that if you changed the install directory in the last step, you have to modified `include_dirs` and `library_dirs` in `setup.py`.

* To use `quartz` library in python, you should make sure the directory where you install `quartz` runtime library, that is `libquartz_runtime.so`, is in python's searching directories.

You can add:

```shell
export LD_LIBRARY_PATH=/your/path/:$LD_LIBRARY_PATH
```

to `~/.bashrc`.

### Dependencies for the simulator

For the simulator, you need to install the HiGHS solver.

```shell
cd external/HiGHS
mkdir build
cd build
cmake ..
make
```

Then add `/your/path/to/quartz/external/HiGHS/build/bin` to `PATH`.
