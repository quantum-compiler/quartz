include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")

link_directories("../..")

file(GLOB_RECURSE WRAPPER_RPC "rpc/wrapper_rpc.cpp")
file(GLOB_RECURSE TEST_RPC "rpc/test_rpc.cpp")
file(GLOB_RECURSE ORACLE "oracle.cpp")

add_library(oracle SHARED ${ORACLE})

add_executable(wrapper_rpc ${WRAPPER_RPC})
add_executable(test_rpc ${TEST_RPC})

target_link_libraries(wrapper_rpc oracle)
target_link_libraries(wrapper_rpc quartz_runtime)

target_link_libraries(test_rpc oracle)
target_link_libraries(test_rpc quartz_runtime)

add_library(librpc STATIC IMPORTED)
set_property(TARGET librpc PROPERTY IMPORTED_LOCATION /home/pengyul/lib/usr/local/lib/librpc.a)
target_link_libraries(wrapper_rpc librpc)
target_link_libraries(test_rpc librpc)