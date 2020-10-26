# ----------------------------------------------------------------------
# General CMake settings
# ----------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
# directory containing additional cmake scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake)

# Use a different project root to find submodule sources
if(DEFINED ENV{PROJECT_ROOT})
  set(PROJECT_ROOT $ENV{PROJECT_ROOT})
else()
  set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})
endif()

# Load commands target_add_dependency, target_add_thirdparty_dependency and target_install
include(${PROJECT_ROOT}/external/project_utils/scripts/cmake/target_add_dependency.cmake)


# ----------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------
project(
  maia VERSION 0.1.0
  DESCRIPTION "Distributed algorithms for CGNS trees"
  LANGUAGES CXX)

## Compiler flags
### Defaults
#include("/scratchm/bberthou/travail/fs_cgns_adapter/fs_cgns_adapter/external/maia/scripts/cmake/default_flags.cmake")
include(${CMAKE_MODULE_PATH}/default_flags.cmake)
set(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} -cpp -DE_DOUBLEREAL -ffixed-line-length-none -fno-second-underscore")
### Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) # default to Release
endif()
### C++ standard
set(CMAKE_CXX_STANDARD          17 )
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON )


# ----------------------------------------------------------------------
# Dependencies 
# ----------------------------------------------------------------------

## Dependencies installed in the environment ##
if(CMAKE_VERSION VERSION_LESS 3.14)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

find_package(MPI REQUIRED)
#find_package(ParaDiGM)
#if(NOT PARADIGM_FOUND)
#  message("ParaDiGM not found please export PARADIGM_HOME to find it")
#endif()
find_package(pdm)

## Dependencies built from source ##
include(FetchContent)
### pybind11 ###
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.6
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
  GIT_TAG        mpi
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
  add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR})
endif()
### doctest ### TODO use this one once the MPI extension has been merged
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
### range-v3 ###
FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG        0.10.0
)
FetchContent_GetProperties(range-v3)
if(NOT range-v3_POPULATED)
  FetchContent_Populate(range-v3)
  add_subdirectory(${range-v3_SOURCE_DIR} ${range-v3_BINARY_DIR})
endif()

## Dependencies from submodules ##
target_add_dependency(${PROJECT_NAME} std_e)
target_add_dependency(${PROJECT_NAME} cpp_cgns)


# ----------------------------------------------------------------------
# Code quality tools
# ----------------------------------------------------------------------
find_program( MEMORYCHECK_COMMAND valgrind )
set( MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )

# > Mandatory to put here to have all shared libs compile with the coverage flags
option(ENABLE_COVERAGE "Enable coverage" OFF)
if(ENABLE_COVERAGE)
  include(CodeCoverage)
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------

## Python, Cython, Pybind11
#include(MixPythonCythonModule)
#mixpython_cython_add_module(maia)
add_subdirectory(maia) # TODO DEL separate file once mixpython_cython_add_module can be included here

## Compiled languages
set(src_dir ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})
set(include_dir ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE all_src_files
  CONFIGURE_DEPENDS "${src_dir}/*.cpp"
)

set(src_files ${all_src_files})
list(FILTER src_files EXCLUDE REGEX ".*\\.test\\.cpp$")
list(FILTER src_files EXCLUDE REGEX ".*\\.pybind\\.cpp$")

add_library(${PROJECT_NAME} SHARED ${src_files})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${include_dir}>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
    MPI::MPI_CXX
    std_e::std_e
    cpp_cgns::cpp_cgns
    range-v3::range-v3
    pdm::pdm_shared
)


# ----------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------
set(SITE_PACKAGES_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/${PROJECT_NAME}")
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

target_install(${PROJECT_NAME})


# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------
include(CTest)
add_subdirectory(test)


# ----------------------------------------------------------------------
# Build documentation
# ----------------------------------------------------------------------
# option(${PROJECT_NAME}_BUILD_DOCUMENTATION "Build ${PROJECT_NAME} documentation ." OFF)
# if(${PROJECT_NAME}_BUILD_DOCUMENTATION)
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
# Summary
# ----------------------------------------------------------------------
include(FeatureSummary)
FEATURE_SUMMARY(WHAT ALL)
