cmake_minimum_required(VERSION 3.18)
project(quantum-superoptimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

file(GLOB_RECURSE TEST_ALL "test/test_all.cpp")
file(GLOB_RECURSE TEST_GENERATOR "test/test_generator.cpp")
file(GLOB_RECURSE SOURCE "context/*" "dag/*" "gate/*" "generator/*" "math/*" "utils/*" "verifier/*")
add_executable(test_all ${TEST_ALL} ${SOURCE})
add_executable(test_generator ${TEST_GENERATOR} ${SOURCE})
