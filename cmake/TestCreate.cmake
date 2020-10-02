################################################################################
#
# test_c_create and test_fortran_create add a Fortran test or a C test
#
# They uses LINK_LIBRARIES and TEST_INC variables
#
################################################################################

# --------------------------------------------------------------------------------
function(seq_test_cpp_create name)
   add_executable(${name} "${name}.cpp")
   if ((NOT MPI_C_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${EXTERNAL_INCLUDES})

   target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
   target_link_libraries(${name} ${CORE_LIBRARIES})
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES})
   target_link_libraries(${name} fmt::fmt)
   target_link_libraries(${name} std_e::std_e)

   install(TARGETS ${name} RUNTIME DESTINATION bin)
   # target_link_libraries(${name} PRIVATE ${LIBRARY_NAME} doctest)
   add_test (${name}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             )
   set_target_properties(${name} PROPERTIES PROCESSOR 1)
endfunction()
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
function(mpi_test_cpp_create name n_proc)
   add_executable(${name} "${name}.cpp")
   # if ((NOT MPI_C_COMPILER) AND MPI_C_COMPILE_FLAGS)
   #   set_target_properties(${name}
   #                         PROPERTIES
   #                         COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
   # endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${MPI_CXX_HEADER_DIR})
   target_include_directories(${name} PRIVATE ${EXTERNAL_INCLUDES})

   # target_link_libraries(${name} ${CORE_LIBRARIES} ${MPI_mpicxx_LIBRARY} ${MPI_mpi_LIBRARY})
   target_link_libraries(${name} ${CORE_LIBRARIES}     )
   target_link_libraries(${name} ${MPI_LIBRARIES}      )
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES} )
   target_link_libraries(${name} fmt::fmt)

   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             ${MPIEXEC_POSTFLAGS})
endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(seq_test_c_create name n_proc)
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

   target_link_libraries(${name} ${LINK_LIBRARIES})
   target_link_libraries(${name} ${CORE_LIBRARIES})
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES})
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name}
             ${CMAKE_CURRENT_BINARY_DIR}/${name})
   set_target_properties(${name} PROPERTIES PROCESSOR ${n_proc})
endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(mpi_test_c_create name n_proc)
   add_executable(${name} "${name}.c")
   if ((NOT MPI_C_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${EXTERNAL_INCLUDES})

   target_link_libraries(${name} ${LINK_LIBRARIES})
   target_link_libraries(${name} ${CORE_LIBRARIES})
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES})
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             ${MPIEXEC_POSTFLAGS})
endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(seq_test_fortran_create name)
   add_executable(${name} "${name}.f90")
   if ((NOT MPI_Fortran_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_Fortran_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   target_link_libraries(${name} ${CORE_LIBRARIES} ${EXTERNAL_LIBRARIES})

   set_target_properties(${name} PROPERTIES LINKER_LANGUAGE "Fortran")
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name}
             ${CMAKE_CURRENT_BINARY_DIR}/${name})
endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(seq_test_fortran77_create name)
   add_executable(${name} "${name}.f")
   if ((NOT MPI_Fortran_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_Fortran_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
   # target_include_directories(${name} PRIVATE ${TEST_INC})

   target_link_libraries(${name} ${CORE_LIBRARIES} ${EXTERNAL_LIBRARIES})

   set_target_properties(${name} PROPERTIES LINKER_LANGUAGE "Fortran")
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name}
             ${CMAKE_CURRENT_BINARY_DIR}/${name})
endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(mpi_test_fortran_create name n_proc)
   add_executable(${name} "${name}.f90")
   if ((NOT MPI_Fortran_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_Fortran_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${MPI_CXX_HEADER_DIR})

   target_link_libraries(${name} ${MPI_Fortran_LIBRARIES})

   set_target_properties(${name} PROPERTIES LINKER_LANGUAGE "Fortran")
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             ${MPIEXEC_POSTFLAGS})
endfunction()
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
function(mpi_test_fortran77_create name n_proc)
   add_executable(${name} "${name}.f")
   if ((NOT MPI_Fortran_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_Fortran_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${MPI_CXX_HEADER_DIR})

   target_link_libraries(${name} ${MPI_Fortran_LIBRARIES})

   set_target_properties(${name} PROPERTIES LINKER_LANGUAGE "Fortran")
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             ${MPIEXEC_POSTFLAGS})
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
function(mpi_test_python_create name n_proc)
  # file (COPY ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}) !! No depndadency CAUTION
  # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.py ${CMAKE_CURRENT_BINARY_DIR}/${name}.py COPYONLY)

  set(output_python_file ${CMAKE_CURRENT_BINARY_DIR}/${name}.py)
  add_custom_command(OUTPUT  "${output_python_file}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                     "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
                     "${output_python_file}"
                     COMMENT "Copying ${name} to the binary directory")

  add_custom_target(tpympi_${name} ALL DEPENDS "${output_python_file}")


  add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
            ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${name}.py
            ${MPIEXEC_POSTFLAGS})

  # set_tests_properties("${name}" PROPERTIES DEPENDS tpympi_${name})
  set_tests_properties("${name}" PROPERTIES
                       ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/mod:$ENV{PYTHONPATH}
                       DEPENDS tpympi_${name})

endfunction()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
function(seq_pytest_python_create name)
  # file (COPY ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.py ${CMAKE_CURRENT_BINARY_DIR}/${name}.py COPYONLY)

  string(REPLACE "/" "_" flat_name ${name} )

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

  add_custom_target(t_${flat_name} ALL DEPENDS "${output_python_file}" "${output_conftest_file}")

  # WORKING_DIRECTORY
  add_test (${flat_name}
            ${Python_EXECUTABLE} -m pytest -r a -v ${output_python_file})

  set_tests_properties("${flat_name}" PROPERTIES
                       ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/mod:$ENV{PYTHONPATH}
                       DEPENDS t_${flat_name})
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

# --------------------------------------------------------------------------------
function(mpi_nec_test_c_create name n_proc)
   add_executable(${name} "${name}.c")
   if ((NOT MPI_C_COMPILER) AND MPI_C_COMPILE_FLAGS)
     set_target_properties(${name}
                           PROPERTIES
                           COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
   endif()
   target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}
                                      PRIVATE ${CMAKE_BINARY_DIR}
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
   # target_include_directories(${name} PRIVATE ${TEST_INC})
   target_include_directories(${name} PRIVATE ${EXTERNAL_INCLUDES})

   target_link_libraries(${name} ${LINK_LIBRARIES})
   target_link_libraries(${name} ${CORE_LIBRARIES})
   target_link_libraries(${name} ${EXTERNAL_LIBRARIES})
   target_link_libraries(${name} ${NEC_AURORA_LIBRARIES})
   install(TARGETS ${name} RUNTIME DESTINATION bin)
   add_test (${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_proc}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_CURRENT_BINARY_DIR}/${name}
             ${MPIEXEC_POSTFLAGS})
endfunction()
# --------------------------------------------------------------------------------
