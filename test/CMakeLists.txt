
# --------------------------------------------------------------------------------
include(TestCreate2)  # New interface - More flexible
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/${PROJECT_NAME})
# message("${SRC_DIR}")
file(GLOB_RECURSE __test_files
     CONFIGURE_DEPENDS
     "${SRC_DIR}/*.test.cpp")
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# foreach(__test_file ${__test_files})
#   message("__test_file::" ${__test_file})
# endforeach()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# > Create unit_test with the proper number of precess
# message("n_proc_max_test::" ${n_proc_max_test})
# mpi_test_create(maia_unit_tests.cpp maia_unit_tests ${n_proc_max_test}
#                 SOURCES   ${__test_files}
#                 INCLUDES  ${CORE_LIBRARIES} #doctest::doctest MPI::MPI_CXX
#                 LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
#                 LABELS     unit
#                 SERIAL_RUN false )
set(n_proc_max_test 4)
mpi_test_create(maia_unit_tests.cpp maia_unit_tests ${n_proc_max_test}
                SOURCES   ${__test_files}
                INCLUDES   # doctest::doctest MPI::MPI_CXX
                LIBRARIES ${LINK_LIBRARIES} MPI::MPI_CXX
                LABELS     unit
                SERIAL_RUN false )
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
mpi_pytest_python_create(maia_python_unit_tests 1)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# set(SRC_DIR ${CMAKE_SOURCE_DIR}/${PROJECT_NAME})
# file(GLOB_RECURSE __py_test_files
#      CONFIGURE_DEPENDS
#      "${SRC_DIR}/test_*.py")
# --------------------------------------------------------------------------------

