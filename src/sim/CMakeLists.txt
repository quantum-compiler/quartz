cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(qsim LANGUAGES C CXX CUDA)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(QSIM_ROOT ${CMAKE_CURRENT_LIST_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")

# Legion to not use static cuda runtime
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

# ##########################################
# custatevec_example build mode
# ##########################################

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build.")
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# custatevec_example dependencies
# ##########################################
set(CUSTATEVEC_ROOT ${CUQUANTUM_ROOT})
if (CUSTATEVEC_ROOT STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUSTATEVEC_ROOT or CUQUANTUM_ROOT to the path of the cuQuantum installation.")
endif ()

message(STATUS "Using CUSTATEVEC_ROOT = ${CUSTATEVEC_ROOT}")

# ##########################################
# custatevec_example global flags
# ##########################################

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND NVCC_FLAGS
  -std=c++17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})

set(CMAKE_CUDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS_ARCH "${CMAKE_CUDA_FLAGS_ARCH_SM70} ${CMAKE_CUDA_FLAGS_ARCH_SM75} ${CMAKE_CUDA_FLAGS_ARCH_SM80} ${CMAKE_CUDA_FLAGS_ARCH_SM90}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_ARCH}")

message(STATUS "cuda compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "nvcc flags: ${CUDA_NVCC_FLAGS}")

set(QSIM_INCLUDE_DIRS "")
set(QSIM_EXT_LIBRARIES "")

include(cuda)

include(cuquantum)

if(USE_LEGION)
  include(legion)
endif()

#if (NCCL_ROOT STREQUAL "")
    #include(nccl)
#else()
list(APPEND QSIM_INCLUDE_DIRS
  ${NCCL_ROOT}/include)
list(APPEND QSIM_EXT_LIBRARIES
  ${NCCL_ROOT}/lib/libnccl${LIBEXT})
#endif()


include(mpi)

list(APPEND QSIM_LD_FLAGS
  -lrt
  -ldl
  -rdynamic)
link_libraries(${QSIM_LD_FLAGS})

list(APPEND QSIM_INCLUDE_DIRS
  ${QSIM_ROOT}/include)

file(GLOB_RECURSE QSIM_HDR
  LIST_DIRECTORIES False
  ${QSIM_ROOT}/include/*.h)

if(USE_LEGION)
  file(GLOB_RECURSE QSIM_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/*.cc)

  file(GLOB_RECURSE QSIM_GPU_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/*.cu)
else()
  file(GLOB_RECURSE QSIM_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/base/*.cc)

  file(GLOB_RECURSE QSIM_GPU_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/base/*.cu)
endif()


cuda_add_library(quansim SHARED ${QSIM_GPU_SRC} ${QSIM_SRC})
set_property(TARGET quansim PROPERTY CUDA_ARCHITECTURES "70;75;80;90")
target_include_directories(quansim PUBLIC ${QSIM_INCLUDE_DIRS})
# TODO: add option to choose simulator backend
if(USE_LEGION)
  target_link_libraries(quansim ${LEGION_LIBRARY} quartz_runtime)
else()
  target_link_libraries(quansim quartz_runtime)
endif()

set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${QSIM_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS quansim DESTINATION ${LIB_DEST})

if(USE_LEGION)
  add_subdirectory(examples/legion-based)
else()
  add_subdirectory(examples/mpi-based)
endif()

