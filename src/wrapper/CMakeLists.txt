include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")

link_directories("../..")

file(GLOB_RECURSE WRAPPER "test_wrapper.cpp")
file(GLOB_RECURSE ORACLE "oracle.cpp")

add_library(oracle SHARED ${ORACLE})

add_executable(wrapper ${WRAPPER})
target_link_libraries(wrapper oracle)
target_link_libraries(wrapper quartz_runtime)
