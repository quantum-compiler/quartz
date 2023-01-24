find_package(MPI REQUIRED)
if(MPI_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES 
    ${MPI_LIBRARIES})
  list(APPEND QSIM_INCLUDE_DIRS
    ${MPI_INCLUDE_DIRS})
  message( STATUS "MPI libraries : ${MPI_LIBRARIES}" )
else()
  message( FATAL_ERROR "MPI not found")
endif()