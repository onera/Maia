# define minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

# ----------------------------------------------------------------------
# project name and supported language
project(
  maia VERSION 0.1.0
  DESCRIPTION "pyCGNS module for distributed algorithms "
  LANGUAGES Fortran CXX C)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Define a short name
set(SHN maia)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# use a different project root to find submodule sources
if(DEFINED ENV{PROJECT_ROOT})
  set(PROJECT_ROOT $ENV{PROJECT_ROOT})
else()
  set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})
endif()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Disallow in-source build
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "maia requires an out-of-source build. Please create a separate "
    "binary directory and run CMake there.")
endif ()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# require C++17
set(CMAKE_CXX_STANDARD          17 )
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON )
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# directory containing UseCython.cmake and FindCython.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# we default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# for testing we will need the python interpreter
if(CMAKE_VERSION VERSION_LESS 3.14)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

find_package(MPI          REQUIRED)
find_package(ParaDiGM)
if(NOT PARADIGM_FOUND)
  message("ParaDiGM not found please export PARADIGM_HOME to find it")
endif()
# find_package(pybind11)
include(CTest)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
find_program( MEMORYCHECK_COMMAND valgrind )
set( MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
## Dependencies ##
include(FetchContent)
### pybind11 ###
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.5.0
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
### doctest mpi ###
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/BerengerBerthoul/doctest.git
  GIT_TAG        f13a00607463ce211004be23b5aa0bdf24d3ab61
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
  add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR})
endif()
### doctest ###
# FetchContent_Declare(
#   doctest
#   GIT_REPOSITORY https://github.com/onqtam/doctest.git
#   GIT_TAG        2.4.0
# )
# FetchContent_GetProperties(doctest)
# if(NOT doctest_POPULATED)
#   FetchContent_Populate(doctest)
#   add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR})
# endif()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# pybind11_add_module(first_step ${CMAKE_SOURCE_DIR}/first_step.cpp)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Default compiler flags
# ----------------------------------------------------------------------
include(${CMAKE_MODULE_PATH}/default_flags.cmake)
# set(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} -cpp -DE_DOUBLEREAL -ffixed-line-length-none -fno-second-underscore")

# ----------------------------------------------------------------------
# > Mandatory to put here to have all shared libs compile with the coverage flags
option(ENABLE_COVERAGE "Enable coverage" OFF)
if(ENABLE_COVERAGE)
  include(CodeCoverage)
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
add_subdirectory(${PROJECT_ROOT}/external/std_e    ${CMAKE_BINARY_DIR}/external/std_e   )
add_subdirectory(${PROJECT_ROOT}/external/cpp_cgns ${CMAKE_BINARY_DIR}/external/cpp_cgns)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# > Build up a single list for all external header / library
set(EXTERNAL_INCLUDES )
set(EXTERNAL_LIBRARIES)
list(APPEND EXTERNAL_INCLUDES  ${MKL_INCLUDE_DIRS}
                               ${MKL_MODULE_PATH}
                               ${PARADIGM_INCLUDE_DIR}
                               ${Python_INCLUDE_DIRS}
                               ${PAPI_INCLUDE_DIR}
                               ${HWLOC_INCLUDE_DIR}
)
if(${CMAKE_VERSION} VERSION_GREATER 3.14)
list(APPEND EXTERNAL_INCLUDES ${Python_NumPy_INCLUDE_DIRS})
endif()

list(APPEND EXTERNAL_LIBRARIES ${BLAS_blas_LIBRARY}
                               ${LAPACK_lapack_LIBRARY}
                               ${MKL_LIBRARIES}
                               ${PARADIGM_LIBRARIES}
                               ${Python_LIBRARIES}
                               ${HWLOC_LIBRARIES}
                               ${PAPI_LIBRARIES}
)
# message("MKL_LIBRARIES$ : " {MKL_LIBRARIES})
# message("EXTERNAL_LIBRARIES$ : " ${EXTERNAL_LIBRARIES})
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
set(MPI_LIBRARIES)
list(APPEND MPI_LIBRARIES ${MPI_mpicxx_LIBRARY} ${MPI_mpi_LIBRARY})
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# this defines cython_add_module
include(UseCython)
# include(UseAURORA)
include(MixPythonCythonModule)
include(SharedModule)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Define installation directory form
set(SITE_PACKAGES_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/${PROJECT_NAME}")
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Compilation of src
add_subdirectory(maia)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
message("${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/")
install(DIRECTORY      "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/" # source directory
        DESTINATION    "include/${PROJECT_NAME}"              # target directory
        FILES_MATCHING                                        # install only matched files
        PATTERN        "*.hpp"                                # select header files
        PATTERN        "*.h"
        )
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# add_subdirectory(mod)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# turn on testing
# enable_testing()
# allow to report to a cdash dashboard
# include(CTest)
# add_subdirectory(tests)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# option(${SHN}_BUILD_DOCUMENTATION "Build ${SHN} documentation ." OFF)
# if(${SHN}_BUILD_DOCUMENTATION)
#   # message("Build docs ... ")

#   include(UseBreathe)

#   add_breathe_doc(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/docs
#                   BUILD_DIR   ${CMAKE_CURRENT_BINARY_DIR}/_build
#                   CACHE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/_doctrees
#                   HTML_DIR    ${CMAKE_CURRENT_BINARY_DIR}/html
#                   DOXY_FILE   ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
#                   CONF_FILE   ${CMAKE_CURRENT_SOURCE_DIR}/docs/conf.py.in
#                   TARGET_NAME maia_doc
#                   ENV_PATH    ${CMAKE_BINARY_DIR}
#                   COMMENT      "HTML documentation"
#                   )

#   install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/html
#           DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc)
# endif()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
include(FeatureSummary)
FEATURE_SUMMARY(WHAT ALL)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# install(FILES       ${CMAKE_CURRENT_LIST_DIR}/mod/gzoumlib/html/gzoumGraph.spec
#         DESTINATION ${SITE_PACKAGES_OUTPUT_DIRECTORY}/gzoumlib/html/)
# ----------------------------------------------------------------------

