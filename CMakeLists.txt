cmake_minimum_required(VERSION 3.16)
project(Quartz)

SET(CMAKE_BUILD_TYPE "Release")

# python
if (MSVC)
    # Could not find the "Development" component on Windows. Using the following commands
    # to get Python_INCLUDE_DIRS and Python_LIBRARY_DIRS.
    find_package(Python COMPONENTS Interpreter)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
            OUTPUT_VARIABLE Python_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import distutils.command.build_ext;
d = distutils.core.Distribution();
b = distutils.command.build_ext.build_ext(d);
b.finalize_options();
print(*b.library_dirs, sep=';')"
            OUTPUT_VARIABLE Python_LIBRARY_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    find_package(Python COMPONENTS Interpreter Development)
endif()
message(STATUS "Python include directories: ${Python_INCLUDE_DIRS}")
include_directories(${Python_INCLUDE_DIRS})
message(STATUS "Python library directories: ${Python_LIBRARY_DIRS}")
link_directories(${Python_LIBRARY_DIRS})

# pybind11
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
    include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    endif()
endif()

# Put the binary files in the same folder as dll files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(USE_ARBLIB FALSE)

if(USE_ARBLIB)
    macro (add_executable _name)
        # invoke built-in add_executable
        _add_executable(${ARGV})
        if (TARGET ${_name})
            target_link_libraries(${_name} flint flint-arb gmp)
        endif()
    endmacro()
    add_compile_definitions(USE_ARBLIB)
endif()

include_directories(${CMAKE_INCLUDE_PATH})
include_directories("src/quartz/")

#initial variables
set(QUARTZ_LIBS "")
set(QUARTZ_LINK_LIBS ${CMAKE_DL_LIBS} pybind11::embed)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE QUARTZ_SRCS
        src/quartz/*.cpp
        )

#Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
    set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
endif()

# if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Build in Debug mode")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "/Od /Wall ${CMAKE_CXX_FLAGS}")
    else ()
        set(CMAKE_CXX_FLAGS "-O0 -g -Wall ${CMAKE_CXX_FLAGS}")
    endif ()
else ()
    if (MSVC)
        set(CMAKE_CXX_FLAGS "/O3 /Wall ${CMAKE_CXX_FLAGS}")
    else ()
        set(CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}")
    endif ()
endif ()

add_library(quartz_runtime SHARED ${QUARTZ_SRCS})

target_compile_features(quartz_runtime PUBLIC cxx_std_17)

target_link_libraries(quartz_runtime ${QUARTZ_LINK_LIBS})

target_include_directories(quartz_runtime
        PUBLIC ${PROJECT_SOURCE_DIR}/src)

execute_process(COMMAND cat /proc/version RESULT_VARIABLE NOT_LINUX)
# TODO: check different OS in a better way here
if (NOT_LINUX)
    if (NOT MSVC)
        message("build on macOS, link openmp lib with compiler clang")
        include_directories(/usr/local/include)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
        find_library(OPENMP_LIB libomp.dylib /usr/local/lib)
        target_link_libraries(quartz_runtime LINK_PUBLIC ${OPENMP_LIB})
    endif()
else()
    message("build on Linux")
    if (DEFINED ENV{CC} AND DEFINED ENV{CXX})
        if ($ENV{CC} MATCHES "icc" AND $ENV{CXX} MATCHES "icc")
            message("build with Intel Compiler icc")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        else()
            message("build with default compiler")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        endif()
    else()
        message("build with default compiler")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

#install library
install(TARGETS quartz_runtime
        LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/src/quartz/
        DESTINATION include/quartz
        FILES_MATCHING PATTERN "*.h")

add_subdirectory(src/test)
add_subdirectory(src/benchmark)

if(QUARTZ_USE_SIMULATOR)
  add_subdirectory(src/sim)
endif()
