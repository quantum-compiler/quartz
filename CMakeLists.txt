cmake_minimum_required(VERSION 3.18)
project(quantum-superoptimization)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE TESTS_SOURCE "test/*.cpp")
file(GLOB_RECURSE SOURCE "context/*" "dag/*" "gate/*" "generator/*" "math/*" "utils/*" "verifier/*")
add_executable(test_all ${TESTS_SOURCE} ${SOURCE})
