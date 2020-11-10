include(TestCreate)

set(pytest_mpi_check_file ${PROJECT_ROOT}/external/pytest-mpi-check/pytest_mpi_check.py)
set(pytest_file_in_build ${CMAKE_CURRENT_BINARY_DIR}/../../pytest-mpi-check/pytest_mpi_check.py)
add_custom_command(OUTPUT  "${pytest_file_in_build}"
                   DEPENDS "${pytest_mpi_check_file}"
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                   "${pytest_mpi_check_file}"
                   "${pytest_file_in_build}"
                   COMMENT "Copying ${pytest_mpi_check_file} to the binary directory")
add_custom_target(pytest_mpi_check ALL DEPENDS "${pytest_file_in_build}")


set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/../${PROJECT_NAME})
file(GLOB_RECURSE test_files
  CONFIGURE_DEPENDS "${src_dir}/*.test.cpp"
)

set(n_proc_max_test 4)
create_mpi_doctest(
  maia
  ${n_proc_max_test}
  SOURCES    maia_unit_tests.cpp ${test_files}
  LIBRARIES  ${LINK_LIBRARIES} MPI::MPI_CXX
  LABEL      unit_tests
  SERIAL_RUN false
)

create_mpi_pytest(maia_python_mpi_tests 1)

option(MAIA_ENABLE_MPI_TESTS "Make CTest run the parallel tests" ON)


if(MAIA_ENABLE_MPI_TESTS)
  mpi_pytest_directory_python_create(maia_python_unit_tests 1)

  # TODO (not ready yet)
  #mpi_test_create(maia_mpi_scheduler_T1.cpp maia_mpi_scheduler_T1 2
  #                SOURCES
  #                LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
  #                LABELS     unit
  #                SERIAL_RUN false )
  #mpi_test_create(maia_mpi_scheduler_threads.cpp maia_mpi_scheduler_threads 15
  #                SOURCES
  #                LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
  #                LABELS     unit
  #                SERIAL_RUN false )
endif()
