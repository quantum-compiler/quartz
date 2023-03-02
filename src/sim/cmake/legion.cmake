# Check availability of precompiled Legion library

# Build Legion from source
message(STATUS "Building Legion from source")
if(FF_USE_PYTHON)
  set(Legion_USE_Python ON CACHE BOOL "enable Legion_USE_Python")
endif()
if(FF_USE_GASNET)
  set(Legion_EMBED_GASNet ON CACHE BOOL "Use embed GASNet")
  set(Legion_EMBED_GASNet_VERSION "GASNet-2022.3.0" CACHE STRING "GASNet version")
  set(Legion_NETWORKS "gasnetex" CACHE STRING "GASNet conduit")
  set(GASNet_CONDUIT ${FF_GASNET_CONDUIT})
endif()
message(STATUS "GASNET ROOT: $ENV{GASNet_ROOT_DIR}")
set(Legion_USE_CUDA ON CACHE BOOL "enable Legion_USE_CUDA" FORCE)
set(Legion_CUDA_ARCH ${FF_CUDA_ARCH} CACHE STRING "Legion CUDA ARCH" FORCE)
add_subdirectory(deps/legion)
set(LEGION_LIBRARY Legion)

set(LEGION_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/legion/runtime)
set(LEGION_DEF_DIR ${CMAKE_BINARY_DIR}/deps/legion/runtime)

#list(APPEND QSIM_INCLUDE_DIRS
#    ${LEGION_INCLUDE_DIR})

