include(${PROJECT_UTILS_CMAKE_DIR}/doctest_pytest.cmake)

# set(pytest_mpi_check_file ${PROJECT_ROOT}/external/pytest-mpi-check/pytest_mpi_check.py)
# set(pytest_file_in_build ${CMAKE_CURRENT_BINARY_DIR}/../external/pytest-mpi-check/pytest_mpi_check.py)
# add_custom_command(OUTPUT  "${pytest_file_in_build}"
#                    DEPENDS "${pytest_mpi_check_file}"
#                    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
#                    "${pytest_mpi_check_file}"
#                    "${pytest_file_in_build}"
#                    COMMENT "Copying ${pytest_mpi_check_file} to the binary directory")
# add_custom_target(pytest_mpi_check ALL DEPENDS "${pytest_file_in_build}")
set(n_proc_max_test 4)


# C++ unit tests
file(GLOB_RECURSE test_files
  CONFIGURE_DEPENDS "${src_dir}/*.test.cpp"
)
create_doctest(
  TESTED_TARGET maia
  LABEL         unit_tests
  SOURCES       maia_unit_tests.cpp ${test_files}
  SERIAL_RUN    false
  N_PROC        ${n_proc_max_test}
)
set_tests_properties(maia_doctest_unit_tests PROPERTIES TIMEOUT 10)


if (${PROJECT_NAME}_ENABLE_MPI_TESTS)
  create_pytest(
    TESTED_FOLDER ${PROJECT_SOURCE_DIR}/maia
    LABEL         unit_tests
    SERIAL_RUN    false
    N_PROC        ${n_proc_max_test}
  )
  create_pytest(
    TESTED_FOLDER ${PROJECT_SOURCE_DIR}/test
    LABEL         func_tests
    SERIAL_RUN    false
    N_PROC        ${n_proc_max_test}
  )

  # TODO (not ready yet)
  #mpi_test_create(maia_mpi_scheduler_T1.cpp maia_mpi_scheduler_T1 maia 2
  #                SOURCES
  #                LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
  #                LABELS     unit
  #                SERIAL_RUN false )
  #mpi_test_create(maia_mpi_scheduler_threads.cpp maia_mpi_scheduler_threads maia 15
  #                SOURCES
  #                LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
  #                LABELS     unit
  #                SERIAL_RUN false )
endif()
