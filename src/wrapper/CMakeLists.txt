include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")

link_directories("../..")

file(GLOB_RECURSE WRAPPER_RPC "rpc/wrapper_rpc.cpp")
file(GLOB_RECURSE TEST_RPC "rpc/test_rpc.cpp")
file(GLOB_RECURSE TEST_ORACLE "test_oracle.cpp")
file(GLOB_RECURSE ORACLE "oracle.cpp")

add_library(oracle SHARED ${ORACLE})
target_link_libraries(oracle quartz_runtime)

add_executable(wrapper_rpc ${WRAPPER_RPC})
add_executable(test_rpc ${TEST_RPC})
add_executable(test_oracle ${TEST_ORACLE})

target_link_libraries(wrapper_rpc oracle)
target_link_libraries(wrapper_rpc quartz_runtime)

target_link_libraries(test_rpc oracle)
target_link_libraries(test_rpc quartz_runtime)

target_link_libraries(test_oracle oracle)
target_link_libraries(test_oracle quartz_runtime)

cmake_path(SET LIBRPC_PATH "$ENV{LIBRPC_PATH}")
cmake_path(GET LIBRPC_PATH PARENT_PATH LIBRPC_PARENT_PATH)
cmake_path(GET LIBRPC_PARENT_PATH PARENT_PATH LIBRPC_PARENT_PARENT_PATH)
cmake_path(SET LIBRPC_INCLUDE_DIR "${LIBRPC_PARENT_PARENT_PATH}/include")
include_directories(${LIBRPC_INCLUDE_DIR})

add_library(librpc STATIC IMPORTED)
set_property(TARGET librpc PROPERTY IMPORTED_LOCATION $ENV{LIBRPC_PATH})
target_link_libraries(wrapper_rpc librpc)
target_link_libraries(test_rpc librpc)
