cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
project(qsim LANGUAGES C CXX CUDA)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(QSIM_ROOT ${CMAKE_CURRENT_LIST_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

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
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS_ARCH "${CMAKE_CUDA_FLAGS_ARCH_SM70} ${CMAKE_CUDA_FLAGS_ARCH_SM75} ${CMAKE_CUDA_FLAGS_ARCH_SM80} ${CMAKE_CUDA_FLAGS_ARCH_SM90}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_ARCH}")

set(QSIM_INCLUDE_DIRS "")
set(QSIM_EXT_LIBRARIES "")

include(cuda)

include(cuquantum)

list(APPEND QSIM_INCLUDE_DIRS
  ${QSIM_ROOT})

file(GLOB_RECURSE QSIM_HDR
  LIST_DIRECTORIES False
  ${QSIM_ROOT}/*.h)

file(GLOB_RECURSE QSIM_SRC
  LIST_DIRECTORIES False
  ${QSIM_ROOT}/*.cc)

file(GLOB_RECURSE QSIM_GPU_SRC
  LIST_DIRECTORIES False
  ${QSIM_ROOT}/*.cu)

cuda_add_library(quansim SHARED ${QSIM_GPU_SRC} ${QSIM_SRC})
set_property(TARGET quansim PROPERTY CUDA_ARCHITECTURES "70;75;80;90")
target_include_directories(quansim PUBLIC ${QSIM_INCLUDE_DIRS})
target_link_libraries(quansim quartz_runtime)

# add_subdirectory(examples)

set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${QSIM_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS quansim DESTINATION ${LIB_DEST})