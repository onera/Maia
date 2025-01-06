include(${PROJECT_UTILS_CMAKE_DIR}/doctest_pytest.cmake)

find_or_fetch_doctest()

# add_custom_target(pytest_parallel ALL DEPENDS "${pytest_file_in_build}")
set(n_proc_max_test 4)


# C++ unit tests
file(GLOB_RECURSE test_files
  CONFIGURE_DEPENDS "${src_dir}/*.test.cpp"
)
create_doctest(
  TESTED_TARGET ${PROJECT_NAME} 
  LABEL         unit_tests
  SOURCES       maia_unit_tests.cpp ${test_files}
  SERIAL_RUN    false
  N_PROC        ${n_proc_max_test}
)
set_tests_properties(maia_doctest_unit_tests PROPERTIES TIMEOUT 60)


create_pytest(
  TESTED_FOLDER ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}
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
