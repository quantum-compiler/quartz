find_package(MPI REQUIRED)
if(MPI_FOUND)
  list(APPEND QSIM_EXT_LIBRARIES
    ${MPI_LIBRARIES})
  list(APPEND QSIM_INCLUDE_DIRS
    ${MPI_CXX_INCLUDE_DIRS})
  message( STATUS "MPI libraries : ${MPI_LIBRARIES}" )
  message( STATUS "MPI include dirs : ${MPI_CXX_INCLUDE_DIRS}" )
  message("QSIM_INCLUDE_DIRS mpi: ${QSIM_INCLUDE_DIRS}")
else()
  message( FATAL_ERROR "MPI not found")
endif()
