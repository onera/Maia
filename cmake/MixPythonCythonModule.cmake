
# ----------------------------------------------------------------------
function( mixpython_cython_add_module _name )
  # message("debug mixpython_cython_add_module")
  # message(${CMAKE_BINARY_DIR})
  # message(${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB_RECURSE _py_files CONFIGURE_DEPENDS *.py)
  set(__py_files)
  foreach(_py_file ${_py_files})
    file(RELATIVE_PATH _py_rel  ${CMAKE_CURRENT_SOURCE_DIR} ${_py_file})
    list(APPEND __py_files  ${_py_rel})
    # message("_py_file :  " ${_py_file})
    # message("_py_rel  :  " ${_py_rel})
  endforeach()

  file(GLOB_RECURSE _pyx_files CONFIGURE_DEPENDS *.pyx)
  foreach(_pyx_file ${_pyx_files})
    # message(${_pyx_file})
    get_filename_component(mod_name ${_pyx_file} NAME_WE )
    # message(${mod_name})

    set_source_files_properties(${_pyx_file} PROPERTIES CYTHON_IS_CXX TRUE)
    # cython_add_module("${_name}_${mod_name}" ${_pyx_file}) // Si 2 modules ont le meme nom
    cython_add_module("${mod_name}" ${_pyx_file})

    # message("core_lib = " ${CORE_LIBRARIES})
    # target_include_directories(${mod_name} PUBLIC ${PROJECT_SOURCE_DIR}/mod)
    target_link_libraries("${mod_name}" ${CORE_LIBRARIES})
    target_link_libraries("${mod_name}" ${EXTERNAL_LIBRARIES})
    if(CMAKE_CUDA_COMPILER)
      target_link_libraries("${mod_name}" ${CUDA_LIBRARIES})
    endif()

    install(TARGETS "${mod_name}"
            LIBRARY DESTINATION ${SITE_PACKAGES_OUTPUT_DIRECTORY}/${rel}/${_name})
  endforeach()

  # Manage install with tree
  file(RELATIVE_PATH rel ${PROJECT_SOURCE_DIR}/mod ${CMAKE_CURRENT_SOURCE_DIR})

  # message("relative path : " ${rel})

  set(python_copied_modules_${_name})
  foreach (python_file IN LISTS __py_files)

      set(output_python_file "${CMAKE_BINARY_DIR}/mod/${rel}/${python_file}")

      add_custom_command(OUTPUT  "${output_python_file}"
                         DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${python_file}"
                         COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                         "${CMAKE_CURRENT_SOURCE_DIR}/${python_file}"
                         "${output_python_file}"
                         COMMENT "Copying ${python_file} to the binary directory")

      get_filename_component(python_file_directory "${python_file}" DIRECTORY)
      get_filename_component(py_file_name_we "${python_file}" NAME_WE)

      # Panic verbose
      # message(" py_file_name_we       " ${py_file_name_we})
      # message(" python_file_directory " ${python_file_directory})
      # message(" output_python_file    " ${output_python_file})
      # message( ${output_python_file})

      set(pyc_file ${output_python_file}c)
      # message( ${Python_EXECUTABLE} -m py_compile ${output_python_file} )
      # message(" outfile pyc : " ${pyc_file})
      add_custom_command(OUTPUT  ${pyc_file}
                         DEPENDS "${output_python_file}"
                         COMMAND ${Python_EXECUTABLE} -m py_compile ${output_python_file})

      # install(FILES       ${pyc_file}
      #         DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}/${rel}/${python_file_directory}"
      #         COMPONENT   "python")
      # list(APPEND python_copied_modules_${_name} "${pyc_file}")

      # Old manner --> Install py file instead
      install(FILES       "${python_file}"
              DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}/${rel}/${python_file_directory}"
              COMPONENT   "python")
      list(APPEND python_copied_modules_${_name} "${output_python_file}")
  endforeach ()

  # message( " copy python  : " "project_python_copy_${_name}")
  add_custom_target(project_python_copy_${_name} ALL
                    DEPENDS
                    ${python_copied_modules_${_name}})


endfunction()
# ----------------------------------------------------------------------
