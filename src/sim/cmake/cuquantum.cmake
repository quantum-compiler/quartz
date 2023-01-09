# find custatevec in CUSTATEVEC_ROOT
find_library(CUSTATEVEC_LIBRARY 
  NAMES libcustatevec${LIBEXT}
  PATHS ${CUSTATEVEC_ROOT}
  PATH_SUFFIXES lib lib64
  DOC "CUSTATEVEC library." )
  
find_path(CUSTATEVEC_INCLUDE_DIR 
    NAMES custatevec.h
    HINTS ${CUSTATEVEC_ROOT}
    PATH_SUFFIXES include 
    DOC "CUSTATEVEC include directory." )

# find custatevec, set custatevec lib and include    
if(CUSTATEVEC_LIBRARY AND CUSTATEVEC_INCLUDE_DIR)
  set(CUSTATEVEC_FOUND ON)
  set(CUSTATEVEC_LIBRARIES ${CUSTATEVEC_LIBRARY})
  set(CUSTATEVEC_INCLUDE_DIRS ${CUSTATEVEC_INCLUDE_DIR})
endif()

# find cuda and custatevec
if(CUSTATEVEC_FOUND)
  list(APPEND QSIM_EXT_LIBRARIES
    ${CUSTATEVEC_LIBRARIES})

  list(APPEND QSIM_INCLUDE_DIRS
    ${CUSTATEVEC_INCLUDE_DIR})
endif()

if(CUSTATEVEC_FOUND)
message( STATUS "CUSTATEVEC inlcude : ${CUSTATEVEC_INCLUDE_DIR}" )
  message( STATUS "CUSTATEVEC libraries : ${CUSTATEVEC_LIBRARIES}" )
  message("QSIM_INCLUDE_DIRS: ${QSIM_INCLUDE_DIRS}")
else()
  message( FATAL_ERROR "CUSTATEVEC package not found -> specify search path via CUQUANTUM_DIR variable")
endif()
