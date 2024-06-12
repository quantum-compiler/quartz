# Quartz Installation

Quartz can be built from source code using the following instructions.

## Install from Source

### Prerequisites

* `apt install build-essential` (Linux/Unix)
* CMAKE 3.16 or higher: `apt install cmake` (Linux/Unix) or `brew install cmake` (MacOS)
  or https://cmake.org/download/ (Windows)
* conda:
    * MacOS:
    ```
    brew install anaconda
    /opt/homebrew/anaconda3/bin/conda init zsh  # Please use your shell name and the directory you installed Anaconda
    ```
    * Other OS: Follow the instructions on https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
* Cython 3.0 or higher (We will install it with Python later by `conda`)
* OpenMP (We will install it later by `conda` or Homebrew)

#### Additional Prerequisites for Windows:

* Microsoft Visual Studio Community 2019 or later (Visual Studio 2022 or later
  recommended): https://visualstudio.microsoft.com/downloads/

### Build Quartz Runtime

1. To get started, clone the Quartz source code from GitHub.

```shell
git clone --recursive https://github.com/quantum-compiler/quartz.git
cd quartz
```

2. Set up the environment (including Python, Cython, OpenMP) by conda.

```shell
conda env create --name quartz --file env.yml
conda activate quartz
conda install openmp  # on MacOS, please run "brew install libomp" instead
```

3. Build the Quartz runtime library (optional with CLion, see [below](INSTALL.md#clion-integration-optional)). This step
   differs a little bit between Windows and other OS.

#### Unix/Linux/MacOS:

```shell
mkdir build
cd build
cmake .. # see notes below
make install
```

Note that line 3 in the example will have the runtime library and include files installed into the default
path `/usr/local/`. To modify the install path, you can set the path explicitly in line 3, for example:

```shell
cmake -D CMAKE_INSTALL_PREFIX:PATH=~/opt/ ..
```

#### Windows:

```batch
mkdir build
cd build
cmake ..
```

Use Visual Studio to open `quartz/build/Quartz.sln`, click Build -> Build Solution (F7).

4. Run Quartz's optimization to see if you installed successfully (optional).

#### Unix/Linux/MacOS:

```shell
cd ..  # should be at the root directory of quartz/ now
./build/test_optimize
```

#### Windows:

```batch
cd ..
:: should be at the root directory of quartz/ now
"build/Debug/test_optimize.exe"
```

You should see an output similar to the following on either OS:

```
number of xfers: 130
[barenco_tof_3] Best cost: 58.000000    candidate number: 22    after 0.170 seconds.
[barenco_tof_3] Best cost: 58.000000    candidate number: 42    after 0.340 seconds.
[barenco_tof_3] Best cost: 58.000000    candidate number: 64    after 0.512 seconds.
...
```

5. Install the Quartz python package. Steps 5 and 6 are optional if you only want to run Quartz in C++.

```shell
cd ../python
python setup.py build_ext --inplace install
```

Note that if you changed the install directory in step 3, you have to modified `include_dirs` and `library_dirs`
in `setup.py`.

6. To use `quartz` library in python, you should make sure the directory where you install `quartz` runtime library,
   that is `libquartz_runtime.so`, is in Python's searching directories.

#### Unix/Linux:

You can add:

```shell
export LD_LIBRARY_PATH=/your/path/:$LD_LIBRARY_PATH
```

to `~/.bashrc`.

## CLion Integration (Optional)

### Additional Prerequisites

* CLion: https://www.jetbrains.com/clion/download/
* First 2 steps in [Build Quartz Runtime](INSTALL.md#build-quartz-runtime)

### CLion Settings

- Settings -> Build, Execution, Deployment -> Toolchains -> add "Visual Studio"
    - Toolset: select the folder where Visual Studio is installed.
    - Architecture: select `amd64`.
    - (Windows) Add environment -> From file -> Environment file: `(path\to\quartz)\scripts\setup_environment.bat`
- Settings -> Build, Execution, Deployment -> CMake -> add "Debug"
- Settings -> Build, Execution, Deployment -> Python Interpreter -> select Python 3.11 (quartz) (the version may vary)
- Settings -> Editor -> Code Style -> Enable ClangFormat
- Settings -> Plugins -> "BlackConnect" -> Install

### Configuration to Run

- `test_optimize | Debug-Visual Studio` -> Edit Configurations... -> Working Directory -> fill in `(path\to\quartz)`
- Click "Run 'test_optimize'" (Shift + F10)

## Visual Studio Integration (Optional) (only for Windows)

### Additional Prerequisites

* Visual Studio: Workloads -> Python Development. You can use Visual Studio Installer to modify an existing Visual
  Studio installation if the Python Development component was not installed before.
* First 3 steps in [Build Quartz Runtime](INSTALL.md#build-quartz-runtime)

### Configuration to Run

- View -> Solution Explorer
- In Solution Explorer, right click `test_optimize`, click "Set as Startup Project"
- In Solution Explorer, right click `test_optimize`, click "Properties"; in the pop-up window, in "Configuration
  Properties", click Debugging -> Working Directory -> dropdown menu -> <Browse...> -> select `(path\to\quartz)`
- Click "Local Windows Debugger/Start Debugging (F5)" or "Start Without Debugging (Ctrl + F5)"

### Troubleshooting

- If there is a pop-up window saying missing "python311.dll" (name varies with Python version) or "zlib.dll", go
  to `(path\to\conda)\envs\quartz`, copy the required .dll files to `(path\to\quartz)\build\Debug`.
    - For any program (test/benchmark) requiring pybind11, please make sure that the working directory of that program
      does **not** contain "python311.dll" (name varies with Python version). If you encounter "Debug Error! ... abort()
      has been called" with Exception 0xe06d7363, this might be the cause.

## Dependencies for the Simulator (Optional)

For the simulator, you need to install the HiGHS solver.

#### Unix/Linux/MacOS:

```shell
cd external/HiGHS
mkdir build
cd build
cmake ..
make
```

Then add `/your/path/to/quartz/external/HiGHS/build/bin` to `PATH`.
