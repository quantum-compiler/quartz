include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")

link_directories("../..")

file(GLOB_RECURSE WRAPPER "wrapper_rpc.cpp")
file(GLOB_RECURSE ORACLE "oracle.cpp")

add_library(oracle SHARED ${ORACLE})

add_executable(wrapper ${WRAPPER})
target_link_libraries(wrapper oracle)
target_link_libraries(wrapper quartz_runtime)

add_library(librpc STATIC IMPORTED)
set_property(TARGET librpc PROPERTY IMPORTED_LOCATION /home/pengyul/lib/usr/local/lib/librpc.a)
target_link_libraries(wrapper librpc)