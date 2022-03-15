# Quartz Installation

Quartz can be built from source code using the following instructions.

## Install from Source

### Prerequisites

* CMAKE 3.16 or higher
* Cython 3.0 or higher

### Build Quartz Runtime

* To get started, clone the Quartz source code from github.

```shell
git clone --recursive https://github.com/quantum-compiler/quartz.git
cd quartz
```

* Set up the environment by using conda

```shell
conda create --name quartz --file env.yml
conda activate quartz
```

The `QUARTZ_HOME` environment is used for building and running Quartz. You can add the following line in `~/.bashrc`.

```shell
export QUARTZ_HOME=/path/to/quartz
```

* Build the Quartz runtime library. The configuration of the Quartz runtime can be modified by `config.cmake`. 

```shell
mkdir build; cd build; cmake ..
sudo make install
```

* Install the Quartz python package.

```shell
cd ../python
python setup.py build_ext --inplace install
```
