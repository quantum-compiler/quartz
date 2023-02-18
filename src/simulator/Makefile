ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_HIP         ?= 0		# Include HIP support (requires HIP)
HIP_TARGET      ?= ROCM
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= simulator
# List all the application source files here
GEN_SRC		?= distributed_simulator.cc distributed_simulator_config.cc	# .cc files
GEN_GPU_SRC	?= distributed_simulator.cu				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I../ -I/usr/local/cuda/include/
CC_FLAGS	?=
NVCC_FLAGS	?=
HIPCC_FLAGS     ?=
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk
