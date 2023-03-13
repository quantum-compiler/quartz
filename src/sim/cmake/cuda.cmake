set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_ROOT ${CUDA_PATH})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
list(APPEND CMAKE_PREFIX_PATH ${CUDA_ROOT})
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  # strip the cudart lib
  string(REGEX REPLACE "[^\;]*cudart[^\;]*(\;?)" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
  set(CUDA_LIBRARIES ${CUDA_LIBRARIES})

  # set cuda runtime and driver lib
  # override cublas and curand because the FindCUDA module may not find the correct libs
  set(CUDADRV_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda${LIBEXT})
  set(CUDA_CUBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas${LIBEXT})
  set(CUDA_curand_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand${LIBEXT})
  #list(APPEND QSIM_EXT_LIBRARIES
    #${CUDADRV_LIBRARIES}
    #${CUDA_CUBLAS_LIBRARIES}
    #${CUDA_curand_LIBRARY})
  list(APPEND QSIM_EXT_LIBRARIES
    ${CUDADRV_LIBRARIES})

  #output
  message( STATUS "CUDA root path : ${CUDA_TOOLKIT_ROOT_DIR}" )
  message( STATUS "CUDA include path : ${CUDA_INCLUDE_DIRS}" )
  message( STATUS "CUDA runtime libraries : ${CUDA_LIBRARIES}" )
  message( STATUS "CUDA driver libraries : ${CUDADRV_LIBRARIES}" )
  message( STATUS "CUBLAS libraries : ${CUDA_CUBLAS_LIBRARIES}" )
  message( STATUS "CURAND libraries : ${CUDA_curand_LIBRARY}" )

  list(APPEND QSIM_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS})

else()
  message( FATAL_ERROR "CUDA package not found -> specify search path via CUDA_ROOT variable")
endif()
