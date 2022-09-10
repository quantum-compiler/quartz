include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")

link_directories("../..")

file(GLOB_RECURSE NAM_SMALL_CIRCUITS "nam_small_circuits.cpp")
add_executable(benchmark_nam_small_circuits ${NAM_SMALL_CIRCUITS})
file(GLOB_RECURSE NAM_MIDDLE_CIRCUITS "nam_middle_circuits.cpp")
add_executable(benchmark_nam_middle_circuits ${NAM_MIDDLE_CIRCUITS})
