# ----------------------------------------------------------------------
# > Automatic deduce file to COMPILE
function( add_auto_shared_module _name )
  # message("debug add_auto_shared_module")
  # message(${CMAKE_BINARY_DIR})
  # message(${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB_RECURSE __sources_files
       CONFIGURE_DEPENDS
       RELATIVE     ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.f90 *.f *.for *.c *.C *.cu)

  list(FILTER __sources_files EXCLUDE REGEX ".*\.test\.cpp$")
  list(FILTER __sources_files EXCLUDE REGEX ".*\.nec\.cpp$")
  list(FILTER __sources_files EXCLUDE REGEX ".*\.in\.for$")

  # foreach( _file ${__sources_files} )
  #   message(" Parse file : " ${_file})
  # endforeach()

  if(AURORA_FOUND)
    nec_add_module(${_name}_nec)
  endif()

  # Create library
  add_library(${_name} SHARED  ${__sources_files})

  # Add include for all libraries configurated
  target_include_directories(${_name} PUBLIC ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(${_name} SYSTEM PUBLIC ${MPI_CXX_HEADER_DIR})
  target_include_directories(${_name} SYSTEM PUBLIC ${EXTERNAL_INCLUDES})
  # target_include_directories(${_name} PRIVATE ${Python_NumPy_INCLUDE_DIRS})

  if(AURORA_FOUND)
    target_include_directories(${_name} PUBLIC ${NEC_AURORA_INCLUDE_DIR})
    target_link_libraries(${_name} ${NEC_AURORA_LIBRARIES})
  endif()

  # if(AURORA_FOUND)
  target_include_directories(${_name} PUBLIC ${THRUST_INCLUDE_DIRS})
  # endif()

  # Link with all libraries configurated
  target_link_libraries(${_name} ${MPI_mpi_LIBRARY} ${MPI_mpifort_LIBRARY} )
  target_link_libraries(${_name} ${EXTERNAL_LIBRARIES})
  # if(CMAKE_CUDA_COMPILER)
  #   target_link_libraries(${_name} ${CUDA_LIBRARIES})
  # endif()
  target_link_libraries(${_name} fmt::fmt)
  target_link_libraries(${_name} std_e::std_e)
  target_link_libraries(${_name} cpp_cgns::cpp_cgns)

  install(TARGETS                   ${_name}
          RUNTIME DESTINATION       bin
          LIBRARY DESTINATION       lib
          PUBLIC_HEADER DESTINATION include
          ARCHIVE DESTINATION       lib)

  install(DIRECTORY      "${CMAKE_CURRENT_SOURCE_DIR}" # source directory
          DESTINATION    "include"                     # target directory
          FILES_MATCHING                               # install only matched files
          PATTERN        "*.hpp"                       # select header files
          PATTERN        "*.h"
          )

endfunction()
# ----------------------------------------------------------------------

## ----------------------------------------------------------------------
## > ARGN have a list of file to compile
#function( add_shared_module _name )
#
#  # foreach( _file ${ARGN} )
#  #   message(" Parse file : " ${_file})
#  # endforeach()
#
#  # Create library
#  add_library(${_name} SHARED ${ARGN})
#
#  # Add include for all libraries configurated
#  target_include_directories(${_name} PUBLIC ${PROJECT_SOURCE_DIR}/src)
#  target_include_directories(${_name} PUBLIC ${MPI_CXX_HEADER_DIR})
#  target_include_directories(${_name} PUBLIC ${EXTERNAL_INCLUDES})
#  # target_include_directories(${_name} PRIVATE Python::NumPy)
#
#  # Link with all libraries configurated
#  target_link_libraries(${_name} ${MPI_mpi_LIBRARY} ${MPI_mpifort_LIBRARY} )
#  target_link_libraries(${_name} ${EXTERNAL_LIBRARIES})
#  target_link_libraries(${_name} fmt::fmt)
#
#  install(TARGETS                   ${_name}
#          RUNTIME DESTINATION       bin
#          LIBRARY DESTINATION       lib
#          PUBLIC_HEADER DESTINATION include
#          ARCHIVE DESTINATION       lib)
#
#  install(DIRECTORY      "${CMAKE_CURRENT_SOURCE_DIR}" # source directory
#          DESTINATION    "include"                     # target directory
#          FILES_MATCHING                               # install only matched files
#          PATTERN        "*.hpp"                       # select header files
#          PATTERN        "*.h"
#          )
#
#endfunction()
## ----------------------------------------------------------------------
