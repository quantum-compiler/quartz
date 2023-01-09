cmake_minimum_required(VERSION 3.10)

project(QSIM_Test)
set(project_target test)

set(CPU_SRC
  test_sim.cc)

cuda_add_executable(${project_target} ${CPU_SRC})
target_include_directories(${project_target} PRIVATE ${QSIM_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
message("QSIM_INCLUDE_DIRS: ${QSIM_INCLUDE_DIRS}")
target_link_libraries(${project_target} -Wl,--whole-archive quansim -Wl,--no-whole-archive ${QSIM_EXT_LIBRARIES})

set(BIN_DEST "bin")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})