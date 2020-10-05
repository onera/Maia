################################################################################
#
# test_c_create and test_fortran_create add a Fortran test or a C test
#
# They uses LINK_LIBRARIES and TEST_INC variables
#
################################################################################

# --------------------------------------------------------------------------------
function(mpi_test_create target_file name n_proc )
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES INCLUDES LIBRARIES LABELS SERIAL_RUN)
  cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # message("${target_file}")
  # foreach(fs ${ARGS_SOURCES})
  #   message("file_source = " ${fs})
  # endforeach(fs)

  add_executable(${name} ${target_file} ${ARGS_SOURCES})

  # > Not working if taget specfied
  # target_include_directories(${name} PRIVATE ${ARGS_INCLUDES})
  # foreach(incl_lib ${ARGS_INCLUDES})
  #   message("${incl_lib}")
  #   target_include_directories(${name} PRIVATE incl_lib)
  # endforeach(incl_lib)
  # target_link_libraries(${name} PUBLIC ${ARGS_LIBRARIES})
  target_link_libraries(${name} maia::maia MPI::MPI_CXX doctest::doctest)

  install(TARGETS ${name} RUNTIME DESTINATION bin)
  add_test (NAME ${name}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
                    ${MPIEXEC_PREFLAGS}
                    ${CMAKE_CURRENT_BINARY_DIR}/${name}
                    ${MPIEXEC_POSTFLAGS})
  # add_test (NAME ${name}
  #           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${name})

  # > Set properties for the current test
  set_tests_properties(${name} PROPERTIES LABELS "${ARGS_LABELS}")
  set_tests_properties(${name} PROPERTIES PROCESSORS nproc)
  if(${ARGS_SERIAL_RUN})
    set_tests_properties(${name} PROPERTIES RUN_SERIAL true)
  endif()
  # > Fail in non slurm
  # set_tests_properties(${name} PROPERTIES PROCESSOR_AFFINITY true)

  # > Specific environement :
  # set_tests_properties(${name} PROPERTIES ENVIRONMENT I_MPI_DEBUG=5)

endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(mpi_pytest_python_create name n_proc)
  # message("seq_pytest_python_create  " "${name}")
  # string(REPLACE "/" "_" flat_name ${name} )
  # message("seq_pytest_python_create  " "${flat_name}")

  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES LABELS SERIAL_RUN)
  cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(output_python_file ${CMAKE_CURRENT_BINARY_DIR}/${name}.py)
  add_custom_command(OUTPUT  "${output_python_file}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                     "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     "${output_python_file}"
                     COMMENT "Copying ${name} to the binary directory")

  set(output_conftest_file ${CMAKE_CURRENT_BINARY_DIR}/conftest.py)
  add_custom_command(OUTPUT  "${output_conftest_file}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py"
                     COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                     "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py"
                     "${output_conftest_file}"
                     COMMENT "Copying conftest.py to the binary directory")

  add_custom_target(t_${name} ALL DEPENDS "${output_python_file}" "${output_conftest_file}")

  # WORKING_DIRECTORY
  add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
            ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} -m pytest -Wignore -r a -v -s ${output_python_file}
            ${MPIEXEC_POSTFLAGS})

  # > Set properties for the current test
  # pytest test/maia_python_unit_tests.py --html=test.html --self-contained-html
  # message(${CMAKE_BINARY_DIR}/maia:$ENV{PYTHONPATH})
  set_tests_properties(${name} PROPERTIES LABELS "${ARGS_LABELS}")
  set_tests_properties("${name}" PROPERTIES
                       ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/:$ENV{PYTHONPATH}
                       DEPENDS t_${name})
  # > Append other
  set_property(TEST "${name}" APPEND PROPERTY
                       ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/maia:$ENV{LD_LIBRARY_PATH})
  set_tests_properties(${name} PROPERTIES PROCESSORS n_proc)
  if(${ARGS_SERIAL_RUN})
    set_tests_properties(${name} PROPERTIES RUN_SERIAL true)
  endif()
  # set_tests_properties(${name} PROPERTIES PROCESSOR_AFFINITY true)


endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(seq_test_python_create name)
  # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.py ${CMAKE_CURRENT_BINARY_DIR}/${name}.py COPYONLY)
  set(output_python_file ${CMAKE_CURRENT_BINARY_DIR}/${name}.py)
  add_custom_command(OUTPUT  "${output_python_file}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                     "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     "${output_python_file}"
                     COMMENT "Copying ${name} to the binary directory")

  add_custom_target(tpyseq_${name} ALL DEPENDS "${output_python_file}")


  add_test (${name}
             ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${name}.py)
  set_tests_properties("${name}" PROPERTIES
                       ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/mod:$ENV{PYTHONPATH}
                       DEPENDS tpyseq_${name})

endfunction()
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
function(seq_nec_test_c_create name)
   add_executable(${name} "${name}.c")
   if ((NOT MPI_C_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   target_include_directories(${name} PRIVATE ${EXTERNAL_INCLUDES})
   target_include_directories(${name} PRIVATE ${NEC_AURORA_INCLUDE_DIR})

   target_link_libraries(${name} ${LINK_LIBRARIES})
   target_link_libraries(${name} ${CORE_LIBRARIES})
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES})
   target_link_libraries(${name} ${NEC_AURORA_LIBRARIES})
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name}
             ${CMAKE_CURRENT_BINARY_DIR}/${name})
endfunction()
# --------------------------------------------------------------------------------
