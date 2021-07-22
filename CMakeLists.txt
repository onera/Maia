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
### C++ standard
set(CMAKE_CXX_STANDARD          17 )
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON )
### fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
### Compiler-dependent flags
set(LOCAL_CMAKE_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake)
include(${LOCAL_CMAKE_SCRIPTS}/compiler_flags.cmake)
### Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) # default to Release
endif()


# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

## Dependencies installed in the environment ##
if(CMAKE_VERSION VERSION_LESS 3.14)
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
  # TODO
  #find_package(Python 3 REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

find_package(MPI REQUIRED)
find_package(Mpi4Py REQUIRED)


#option(MAIA_USE_PDM_INSTALL OFF "Use an installed ParaDiGM version")
#if (MAIA_USE_PDM_INSTALL)
#  find_package(pdm CONFIG REQUIRED) # "CONFIG" require the install to have a pdmConfig.cmake
#  if(NOT pdm_FOUND)
#    message("ParaDiGM not found please export pdm_DIR to find it")
#  endif()
#else()


find_package(pdma CONFIG)
if(NOT pdma_FOUND)
  find_package(pdm  CONFIG)
  if(NOT pdm_FOUND)
    check_sub_repo_exists(paradigm)

    # Maia uses ParaDiGM with these options
    set(PDM_ENABLE_Fortran         OFF CACHE BOOL "Maia uses ParaDiGM without Fortran"     FORCE)
    set(PDM_ENABLE_STATIC          OFF CACHE BOOL "Maia uses ParaDiGM without static libs" FORCE)
    set(PDM_ENABLE_SHARED          ON  CACHE BOOL "Maia uses ParaDiGM with shared libs"    FORCE)
    set(PDM_ENABLE_PYTHON_BINDINGS ON  CACHE BOOL "Maia uses ParaDiGM with python"         FORCE)
    set(PDM_ENABLE_PARMETIS        ON  CACHE BOOL "Maia uses ParaDiGM without parmetis"    FORCE)
    set(PDM_ENABLE_PTSCOTCH        ON  CACHE BOOL "Maia uses ParaDiGM with Scotch"         FORCE)
    set(PDM_ENABLE_MPI_CHECK       OFF CACHE BOOL "Maia uses ParaDiGM without MPI check"   FORCE)

    # ParaDiGM unit tests if Maia unit tests
    if(MAIA_ENABLE_MPI_TESTS)
      set(PDM_ENABLE_UNIT_TEST ON  CACHE BOOL "Maia uses ParaDiGM with unit tests"     FORCE)
    else()
      set(PDM_ENABLE_UNIT_TEST OFF CACHE BOOL "Maia uses ParaDiGM without unit tests"  FORCE)
    endif()

    target_add_dependency(${PROJECT_NAME} paradigm)
  endif()
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
include(${PROJECT_UTILS_CMAKE_DIR}/target_check_local_dependency.cmake)
target_check_local_dependency(std_e OFF) # OFF means not required, TODO make arg optional
set(STD_E_ENABLE_MPI ON CACHE BOOL "Maia requires MPI" FORCE)
if (std_e_FOUND)
  target_add_dependency(${PROJECT_NAME} std_e)
else()
  # TODO fix if std_e not compiled here, doctest is not brought to cmake
  include(FetchContent)
  ### doctest ###
  FetchContent_Declare(
    doctest
    #GIT_REPOSITORY https://github.com/onqtam/doctest.git
    GIT_REPOSITORY https://github.com/BerengerBerthoul/doctest.git
    GIT_TAG        f13a00607463ce211004be23b5aa0bdf24d3ab61
    #GIT_TAG        mpi
  )
  FetchContent_GetProperties(doctest)
  if(NOT doctest_POPULATED)
    FetchContent_Populate(doctest)
    add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR})
  endif()

  find_package(std_e CONFIG REQUIRED)
endif()

include(${PROJECT_UTILS_CMAKE_DIR}/target_check_local_dependency.cmake)
target_check_local_dependency(cpp_cgns OFF) # OFF means not required, TODO make arg optional
if (cpp_cgns_FOUND)
  target_add_dependency(${PROJECT_NAME} cpp_cgns)
else()
  # TODO fix if cpp_cgns not compiled here, doctest is not brought to cmake
  include(FetchContent)
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

  find_package(cpp_cgns CONFIG REQUIRED)
endif()

## Python dependencies (just check) ##
include(${PROJECT_UTILS_CMAKE_DIR}/has_python_module.cmake)
#has_python_module(ruamel REQUIRED)
#has_python_module(pytest)
#has_python_module(pytest_mpi) # TODO tell to install package pytest-mpi
#has_python_module(pytest_html) # TODO tell to install package pytest-mpi
#has_python_module(pytest_check)


# ----------------------------------------------------------------------
# Code quality tools
# ----------------------------------------------------------------------
find_program( MEMORYCHECK_COMMAND valgrind )
set( MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )

# > Mandatory to put here to have all shared libs compile with the coverage flags
option(${PROJECT_NAME}_ENABLE_COVERAGE "Enable coverage for ${PROJECT_NAME}" OFF)
#if(${${PROJECT_NAME}_ENABLE_COVERAGE})
 #include(CodeCoverage)
 #APPEND_COVERAGE_COMPILER_FLAGS()
#endif()


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
install_python_modules(${PROJECT_NAME})

# install pytest-mpi-check
install(
  DIRECTORY   "${PROJECT_ROOT}/external/pytest-mpi-check/pytest_mpi_check"
  DESTINATION "${SITE_PACKAGES_OUTPUT_DIRECTORY}"
  COMPONENT   "python"
)

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
  include(${PROJECT_UTILS_CMAKE_DIR}/build_documentation.cmake)
  build_documentation()
endif()


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
get_directory_property(top_lvl_cmakelists PARENT_DIRECTORY)
if(top_lvl_cmakelists STREQUAL ${PROJECT_SOURCE_DIR})
  include(FeatureSummary)
  FEATURE_SUMMARY(WHAT ALL)
endif()
