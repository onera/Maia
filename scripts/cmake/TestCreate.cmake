function(create_mpi_doctest tested_target n_proc)
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES LIBRARIES LABEL SERIAL_RUN)
  cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(executable_for_tests ${tested_target}_${LABEL})
  add_executable(${executable_for_tests} ${ARGS_SOURCES})

  #target_link_libraries(${executable_for_tests} maia::maia MPI::MPI_CXX doctest::doctest)
  target_link_libraries(${executable_for_tests} ${tested_target}::${tested_target} MPI::MPI_CXX doctest::doctest)

  install(TARGETS ${executable_for_tests} RUNTIME DESTINATION bin)
  if(${ARGS_SERIAL_RUN})
    add_test(
      NAME ${executable_for_tests}
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${executable_for_tests}
    )
  else()
    add_test(
      NAME ${executable_for_tests}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
              ${MPIEXEC_PREFLAGS}
              ${CMAKE_CURRENT_BINARY_DIR}/${executable_for_tests}
              ${MPIEXEC_POSTFLAGS}
    )
  endif()

  # > Set properties for the current test
  set_tests_properties(${executable_for_tests} PROPERTIES LABELS "${ARGS_LABEL}")
  set_tests_properties(${executable_for_tests} PROPERTIES PROCESSORS nproc)
  set_tests_properties(${executable_for_tests} PROPERTIES RUN_SERIAL ${ARGS_SERIAL_RUN})

  # > Fail in non slurm
  # set_tests_properties(${target_name} PROPERTIES PROCESSOR_AFFINITY true)

  # > Specific environement :
  # set_tests_properties(${target_name} PROPERTIES ENVIRONMENT I_MPI_DEBUG=5)
endfunction()

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(create_mpi_pytest name n_proc)
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

  # set(output_conftest_file ${CMAKE_CURRENT_BINARY_DIR}/conftest.py)
  # add_custom_command(OUTPUT  "${output_conftest_file}"
  #                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py"
  #                    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
  #                    "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py"
  #                    "${output_conftest_file}"
  #                    COMMENT "Copying conftest.py to the binary directory")
  # add_custom_target(t_${name} ALL DEPENDS "${output_python_file}" "${output_conftest_file}")
  add_custom_target(t_${name} ALL DEPENDS "${output_python_file}" )

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
function(mpi_pytest_directory_python_create name n_proc)
  # message("seq_pytest_python_create  " "${name}")
  # string(REPLACE "/" "_" flat_name ${name} )
  # message("seq_pytest_python_create  " "${flat_name}")
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES LABELS SERIAL_RUN)
  cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # --------------------------------------------------------------------------------
  file(GLOB_RECURSE __py_test_files
       CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  foreach(__py_test_file ${__py_test_files})

    # message("__py_test_file::" ${__py_test_file})

    get_filename_component( output_python_file_name ${__py_test_file} NAME_WE   )
    get_filename_component( output_python_dir       ${__py_test_file} DIRECTORY )
    file(RELATIVE_PATH output_python_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${output_python_dir})

    set(output_python_file ${CMAKE_CURRENT_BINARY_DIR}/${output_python_dir_rel}/${output_python_file_name}.py)
    # message(${output_python_file})
    add_custom_command(OUTPUT  "${output_python_file}"
                       DEPENDS "${__py_test_file}"
                       COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                       "${__py_test_file}"
                       "${output_python_file}"
                       COMMENT "Copying ${output_python_dir_rel}/${output_python_file_name}.py to the binary directory")

    set(rel_output_python_file ${output_python_dir_rel}/${output_python_file_name}.py)
    # message("rel_output_python_file::" ${rel_output_python_file})
    string(REPLACE "/" "_" target_name "${rel_output_python_file}")
    # message("target_name::" ${target_name})
    add_custom_target(t_${target_name} ALL DEPENDS "${output_python_file}")
  endforeach()
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  file(GLOB_RECURSE __conftest_files
       CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*conftest.py")
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  foreach(__conftest_file ${__conftest_files})

    # message("__conftest_file::" ${__conftest_file})

    get_filename_component( output_python_file_name ${__conftest_file} NAME_WE   )
    get_filename_component( output_python_dir       ${__conftest_file} DIRECTORY )
    file(RELATIVE_PATH output_python_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${output_python_dir})

    set(output_python_file ${CMAKE_CURRENT_BINARY_DIR}/${output_python_dir_rel}/${output_python_file_name}.py)
    # message(${output_python_file})
    add_custom_command(OUTPUT  "${output_python_file}"
                       DEPENDS "${__conftest_file}"
                       COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                       "${__conftest_file}"
                       "${output_python_file}"
                       COMMENT "Copying ${output_python_dir_rel}/${output_python_file_name}.py to the binary directory")

    set(rel_output_python_file ${output_python_dir_rel}/${output_python_file_name}.py)
    # message("rel_output_python_file::" ${rel_output_python_file})
    string(REPLACE "/" "_" target_name "${rel_output_python_file}")
    # message("target_name::" ${target_name})
    add_custom_target(t_${target_name} ALL DEPENDS "${output_python_file}")
  endforeach()
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  file(GLOB_RECURSE __pytestini_files
       CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*pytest.ini")
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  foreach(__pytestini_file ${__pytestini_files})

    # message("__pytestini_file::" ${__pytestini_file})

    get_filename_component( output_pytestini_file_name ${__pytestini_file} NAME_WE   )
    get_filename_component( output_pytestini_dir       ${__pytestini_file} DIRECTORY )
    file(RELATIVE_PATH output_pytestini_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${output_pytestini_dir})

    set(output_pytestini_file ${CMAKE_CURRENT_BINARY_DIR}/${output_pytestini_dir_rel}/${output_pytestini_file_name}.ini)
    # message("output_pytestini_file" ${output_pytestini_file})
    add_custom_command(OUTPUT  "${output_pytestini_file}"
                       DEPENDS "${__pytestini_file}"
                       COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                       "${__pytestini_file}"
                       "${output_pytestini_file}"
                       COMMENT "Copying ${output_pytestini_dir_rel}/${output_pytestini_file_name}.ini to the binary directory")
    set(rel_output_pytestini_file ${output_pytestini_dir_rel}/${output_pytestini_file_name}.ini)
    # message("rel_output_pytestini_file::" ${rel_output_pytestini_file})
    string(REPLACE "/" "_" target_name "${rel_output_pytestini_file}")
    # message("target_name::" ${target_name})
    add_custom_target(t_${target_name} ALL DEPENDS "${output_pytestini_file}")
  endforeach()
  # --------------------------------------------------------------------------------

  # WORKING_DIRECTORY
  # add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
  #           ${MPIEXEC_PREFLAGS}
  #           ${Python_EXECUTABLE} -m pytest -Wignore -r a -v -s ${name}
  #           ${MPIEXEC_POSTFLAGS})
  add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
            ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} -m pytest -Wignore -r a -v -s .
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
  # > Not working if not launch with srun ...
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
