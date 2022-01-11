cmake_minimum_required(VERSION 3.16)
project(Quartz)

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")

#initial variables
set(QUARTZ_LIBS "")
set(QUARTZ_LINK_LIBS ${CMAKE_DL_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE QUARTZ_SRCS
  src/core/*.cpp
  )
# list(APPEND TASO_SRCS ${PROTO_CPPS})

#Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
#   set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
#   set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

add_library(quartz_runtime SHARED ${QUARTZ_SRCS})

# set_target_properties(quartz_runtime
#   PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(quartz_runtime PUBLIC cxx_std_17)

target_link_libraries(quartz_runtime ${QUARTZ_LINK_LIBS})

target_include_directories(quartz_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

# set_target_properties(quartz_runtime
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#install library
install(TARGETS quartz_runtime
    LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)
