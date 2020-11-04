include(UseCython)

# ----------------------------------------------------------------------
function( mixpython_cython_add_module _name )
  # message("debug mixpython_cython_add_module")
  # message(${CMAKE_BINARY_DIR})
  # message(${CMAKE_CURRENT_SOURCE_DIR})

# Pybind
  # if(ENABLE_PYBIND)
  file(GLOB_RECURSE _pybind_files CONFIGURE_DEPENDS *.pybind.cpp)
  foreach(_pybind_file ${_pybind_files})
    # message("_pybind_file::" ${_pybind_file})
    # Deduce module name
    get_filename_component(mod_name ${_pybind_file} NAME_WE)
    get_filename_component(pybind_dir ${_pybind_file} DIRECTORY)
    #get_filename_component( pybind_mod_name ${_pybind_file} NAME_WE   )
    file(RELATIVE_PATH pybind_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${pybind_dir})
    message("mod_name::" ${mod_name} ${pybind_dir_rel})

    # > If same name : problem
    pybind11_add_module(${mod_name} ${_pybind_file})
    target_link_libraries(${mod_name} PUBLIC std_e::std_e maia::maia MPI::MPI_CXX) # TODO rm std_e, MPI ? (transitive from maia)
    target_include_directories(${mod_name} PUBLIC ${Mpi4Py_INCLUDE_DIR})
    set_target_properties(${mod_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pybind_dir_rel}")

    install(TARGETS "${mod_name}"
            LIBRARY DESTINATION ${SITE_PACKAGES_OUTPUT_DIRECTORY}/${pybind_dir_rel})
  endforeach()
  # endif()

# Cython
  file(GLOB_RECURSE _pyx_files CONFIGURE_DEPENDS *.pyx)
  foreach(_pyx_file ${_pyx_files})
    # message("pyx_file::" ${_pyx_file})

    # Il faut ici refaire le tag du module avec des underscore au cas ou -_-
    get_filename_component(mod_name ${_pyx_file} NAME_WE )
    # message("mod_name::" ${mod_name})
    # message("rel::" ${rel})

    get_filename_component( pyx_dir      ${_pyx_file} DIRECTORY )
    #get_filename_component( pyx_mod_name ${_pyx_file} NAME_WE   )
    file(RELATIVE_PATH pyx_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${pyx_dir})

    set_source_files_properties(${_pyx_file} PROPERTIES CYTHON_IS_CXX TRUE)
    # cython_add_module("${_name}_${mod_name}" ${_pyx_file}) // Si 2 modules ont le meme nom
    cython_add_one_file_module("${mod_name}" ${_pyx_file})
    set_target_properties(${mod_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pyx_dir_rel}")

    # message("core_lib = " ${CORE_LIBRARIES})
    # target_include_directories(${mod_name} PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries("${mod_name}" ${CORE_LIBRARIES})
    target_link_libraries("${mod_name}" ${EXTERNAL_LIBRARIES})

    install(TARGETS "${mod_name}"
            LIBRARY DESTINATION ${SITE_PACKAGES_OUTPUT_DIRECTORY}/${pyx_dir_rel})
  endforeach()

  # Manage install with tree
  #file(RELATIVE_PATH rel ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  # message("relative path : " ${rel})

# Python
  set(python_copied_modules_${_name})

  file(GLOB_RECURSE _py_files CONFIGURE_DEPENDS *.py)
  foreach (python_file IN LISTS _py_files)

    file(RELATIVE_PATH python_rel_file  ${CMAKE_CURRENT_SOURCE_DIR} ${python_file})
    set(output_python_file "${CMAKE_CURRENT_BINARY_DIR}/${python_rel_file}")

    # message("output_python_file::" ${output_python_file})

    add_custom_command(OUTPUT  "${output_python_file}"
                       DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${python_rel_file}"
                       COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                       "${CMAKE_CURRENT_SOURCE_DIR}/${python_rel_file}"
                       "${output_python_file}"
                       COMMENT "Copying ${python_rel_file} to the binary directory")

    set(pyc_file ${output_python_file}c)
    add_custom_command(OUTPUT  ${pyc_file}
                       DEPENDS "${output_python_file}"
                       COMMAND ${Python_EXECUTABLE} -m py_compile ${output_python_file})


    get_filename_component(python_file_directory "${python_rel_file}" DIRECTORY)
    # install(FILES       ${pyc_file}
    #         DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}/${rel}/${python_file_directory}"
    #         COMPONENT   "python")
    # list(APPEND python_copied_modules_${_name} "${pyc_file}")

    # # Old manner --> Install py file instead
    # install(FILES       "${python_rel_file}"
    #         DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}/${rel}/${python_file_directory}"
    #         COMPONENT   "python")
    # Old manner --> Install py file instead
    install(FILES       "${python_rel_file}"
            DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}/${python_file_directory}"
            COMPONENT   "python")
    list(APPEND python_copied_modules_${_name} "${output_python_file}")
  endforeach ()

  # message( " copy python  : " "project_python_copy_${_name}")
  add_custom_target(project_python_copy_${_name} ALL
                    DEPENDS
                    ${python_copied_modules_${_name}})


endfunction()
# ----------------------------------------------------------------------
