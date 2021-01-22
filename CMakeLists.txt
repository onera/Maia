# ----------------------------------------------------------------------
# General CMake settings
# ----------------------------------------------------------------------
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

if(NOT DEFINED PROJECT_ROOT)
  set(PROJECT_ROOT ${CMAKE_SOURCE_DIR} CACHE PATH "Root directory, where the submodules are populated")
endif()

macro(check_sub_repo_exists sub_repo_name)
  set(sub_repo_path "${PROJECT_ROOT}/external/${sub_repo_name}")
  file(GLOB sub_repo_files ${sub_repo_path}/*)
  list(LENGTH sub_repo_files sub_repo_nb_files)
  if(sub_repo_nb_files EQUAL 0)
    message(FATAL_ERROR
      "${PROJECT_ROOT}/external/${sub_repo_name} is empty.
      Maybe you forgot to initialize it with \"git submodule update --init\""
    )
  endif()
endmacro()

check_sub_repo_exists(project_utils)
set(PROJECT_UTILS_CMAKE_DIR ${PROJECT_ROOT}/external/project_utils/scripts/cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_UTILS_CMAKE_DIR}") # for custom Find*.cmake files
include(${PROJECT_UTILS_CMAKE_DIR}/target_add_dependency.cmake) # target_add_dependency, target_add_thirdparty_dependency and target_install commands

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# ----------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------
project(
  maia VERSION 0.1.0
  DESCRIPTION "Distributed algorithms for CGNS trees"
  LANGUAGES CXX)

## Compiler flags
### Defaults
set(LOCAL_CMAKE_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake)
include(${LOCAL_CMAKE_SCRIPTS}/default_flags.cmake)
### Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) # default to Release
endif()
### C++ standard
set(CMAKE_CXX_STANDARD          17 )
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON )
### fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


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
find_package(Mpi4Py REQUIRED)

option(MAIA_USE_PDM_INSTALL OFF "Use an installed ParaDiGM version")
if (MAIA_USE_PDM_INSTALL)
  find_package(pdm CONFIG REQUIRED) # "CONFIG" require the install to have a pdmConfig.cmake
  if(NOT pdm_FOUND)
    message("ParaDiGM not found please export pdm_DIR to find it")
  endif()
else()
  check_sub_repo_exists(paradigm)

  # Maia uses ParaDiGM with these options
  set(PDM_ENABLE_Fortran         OFF CACHE BOOL "Maia uses ParaDiGM without Fortran"     FORCE)
  set(PDM_ENABLE_STATIC          OFF CACHE BOOL "Maia uses ParaDiGM without static libs" FORCE)
  set(PDM_ENABLE_SHARED          ON  CACHE BOOL "Maia uses ParaDiGM with shared libs"    FORCE)
  set(PDM_ENABLE_PYTHON_BINDINGS ON  CACHE BOOL "Maia uses ParaDiGM with python"         FORCE)
  set(PDM_ENABLE_PARMETIS        OFF CACHE BOOL "Maia uses ParaDiGM without parmetis"    FORCE)
  set(PDM_ENABLE_PTSCOTCH        ON  CACHE BOOL "Maia uses ParaDiGM with Scotch"         FORCE)
  set(PDM_ENABLE_MPI_CHECK       OFF CACHE BOOL "Maia uses ParaDiGM without MPI check"   FORCE)
  set(PDM_ENABLE_MPI_CHECK       OFF CACHE BOOL "Maia uses ParaDiGM without MPI check"   FORCE)

  # ParaDiGM does not use the same variable names as FindPython
  message("Python_EXECUTABLE = ${Python_EXECUTABLE}")
  set(PYTHON_EXECUTABLE  ${Python_EXECUTABLE}   CACHE FILEPATH "" FORCE)
  set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS} CACHE PATH     "" FORCE)
  set(PYTHON_LIBRARY     ${Python_LIBRARIES}    CACHE FILEPATH "" FORCE)

  # ParaDiGM needs these paths from Scotch
  if(NOT DEFINED ENV{PTSCOTCH_DIR})
    message(FATAL_ERROR "You need PtScotch to build ParaDiGM. Export environment variable PTSCOTCH_DIR")
  endif()
  set(PTSCOTCH_DIR       $ENV{PTSCOTCH_DIR}     CACHE PATH "" FORCE)
  set(PTSCOTCH_LIBRARIES "${PTSCOTCH_DIR}/lib/libptscotch.so;${PTSCOTCH_DIR}/lib/libscotch.so;${PTSCOTCH_DIR}/lib/libptscotcherr.so" CACHE STRING "" FORCE)
  #set(PTSCOTCH_VERSION        CACHE STRING FORCE)

  # ParaDiGM unit tests if Maia unit tests
  if(MAIA_ENABLE_MPI_TESTS)
    set(PDM_ENABLE_UNIT_TEST ON  CACHE BOOL "Maia uses ParaDiGM with unit tests"     FORCE)
  else()
    set(PDM_ENABLE_UNIT_TEST OFF CACHE BOOL "Maia uses ParaDiGM without unit tests"  FORCE)
  endif()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # The C compiler does not link libm by default
    set(CMAKE_C_FLAGS "-lm ${CMAKE_C_FLAGS}")
  endif()
  target_add_dependency(${PROJECT_NAME} paradigm)
endif()

## Dependencies built from source ##
include(FetchContent)
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
check_sub_repo_exists(std_e)
set(STD_E_ENABLE_MPI ON CACHE BOOL "Maia requires MPI" FORCE)
target_add_dependency(${PROJECT_NAME} std_e)

check_sub_repo_exists(cpp_cgns)
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
set(src_dir ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})
set(test_dir ${PROJECT_SOURCE_DIR}/test)
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
target_install(${PROJECT_NAME})

# ----------------------------------------------------------------------
# Python and wrapping
# ----------------------------------------------------------------------
include(${PROJECT_UTILS_CMAKE_DIR}/python_wrapping.cmake)
compile_install_pybind_modules(${PROJECT_NAME})
compile_install_python_modules(${PROJECT_NAME})

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------
include(CTest)
add_subdirectory(test)


# ----------------------------------------------------------------------
# Build documentation
# ----------------------------------------------------------------------
option(MAIA_BUILD_DOCUMENTATION "Build ${PROJECT_NAME} documentation" OFF)
if(MAIA_BUILD_DOCUMENTATION)
  # Download bootstrap style
  FetchContent_Declare(
    sphinx_bootstrap_theme
    GIT_REPOSITORY https://github.com/ryan-roemer/sphinx-bootstrap-theme.git
    GIT_TAG        v0.8.0
  )
  FetchContent_GetProperties(sphinx_bootstrap_theme)
  if(NOT sphinx_bootstrap_theme_POPULATED)
    FetchContent_Populate(sphinx_bootstrap_theme)
  endif()

  include(UseBreathe)
  add_breathe_doc(TARGET_NAME ${PROJECT_NAME}_doc
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc
    DOXY_FILE  ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    CONF_FILE  ${CMAKE_CURRENT_SOURCE_DIR}/doc/conf.py.in
    BUILD_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_build
    CACHE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_doctrees
    HTML_DIR   ${CMAKE_CURRENT_BINARY_DIR}/html
    #ENV_PATH    ${CMAKE_BINARY_DIR}
    COMMENT "HTML documentation"
  )

  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/html
          DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc)
endif()


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
get_directory_property(is_top_lvl_cmakelists PARENT_DIRECTORY)
if(is_top_lvl_cmakelists)
  include(FeatureSummary)
  FEATURE_SUMMARY(WHAT ALL)
endif()
