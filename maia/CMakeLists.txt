# ----------------------------------------------------------------------
# this defines cython_add_module
include(UseCython)
include(MixPythonCythonModule)
# ----------------------------------------------------------------------

# --------------------------------------------------------------
mixpython_cython_add_module(maia)
# --------------------------------------------------------------

set(src_dir ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})
set(include_dir ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE all_src_files
  RELATIVE ${src_dir}
  *.cpp *.f90 *.f *.for *.c *.C
)

set(src_files ${all_src_files})
list(FILTER src_files EXCLUDE REGEX ".*\\.test\\.cpp$")
list(FILTER src_files EXCLUDE REGEX ".*\\.in\\.for$")
list(FILTER src_files EXCLUDE REGEX ".*\\.pybind\\.cpp$")

add_library(${PROJECT_NAME} SHARED ${src_files})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
    std_e::std_e
    cpp_cgns::cpp_cgns
    MPI::MPI_CXX
)


# ----------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------
set(SITE_PACKAGES_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/${PROJECT_NAME}")
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

target_install(${PROJECT_NAME})
