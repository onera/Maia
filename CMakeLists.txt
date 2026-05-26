# ------------------------------------------------------------------------------
# General CMake settings
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW) # force find_package to take <PackageName>_ROOT variables into account

if(NOT DEFINED PROJECT_ROOT)
  set(PROJECT_ROOT ${CMAKE_SOURCE_DIR} CACHE PATH "Root directory, where the submodules are populated")
endif()

# Bootstrap project_utils
file(GLOB project_utils_files "${PROJECT_ROOT}/external/project_utils/*")
list(LENGTH project_utils_files n_file_project_utils)
if(n_file_project_utils EQUAL 0)
  message(FATAL_ERROR
    "external/project_utils is empty.
    Maybe you forgot to initialize it with \"git submodule update --init\""
  )
endif()
set(PROJECT_UTILS_DIR ${PROJECT_ROOT}/external/project_utils)
set(PROJECT_UTILS_CMAKE_DIR ${PROJECT_UTILS_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_UTILS_CMAKE_DIR}/find_package") # for custom Find*.cmake files
include(${PROJECT_UTILS_CMAKE_DIR}/dependency_management.cmake) # project_add_subdirectory, project_find_package and target_install...

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------
project(
  maia VERSION 0.1.0
  DESCRIPTION "Distributed algorithms for CGNS trees"
  LANGUAGES CXX
)

option(${PROJECT_NAME}_ENABLE_COVERAGE "Enable coverage for ${PROJECT_NAME}" OFF)
option(${PROJECT_NAME}_ENABLE_DOCUMENTATION "Build ${PROJECT_NAME} documentation" OFF)
option(${PROJECT_NAME}_BUILD_EMBEDDED_PDM "Build the included copy of ParaDiGM" ON)
option(${PROJECT_NAME}_ENABLE_TESTS "Make CTest run the tests" ON)

# Check repo and submodules validity
include(${PROJECT_UTILS_CMAKE_DIR}/check_repo_validity.cmake) # check all submodules are present
if (${PROJECT_NAME}_BUILD_EMBEDDED_PDM)
  set(optional_submodules " ")
else()
  set(optional_submodules "paradigm")
endif()
check_repo_validity(${optional_submodules})

## Compiler flags
### C++ standard
set(SUPPORTED_CXX_STANDARDS 17 20)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to compile against")
endif()
if(NOT CMAKE_CXX_STANDARD IN_LIST SUPPORTED_CXX_STANDARDS)
  message(FATAL_ERROR "Supported CXX standards are: ${SUPPORTED_CXX_STANDARDS}.")
endif()

set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON )
### fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
### Compiler-dependent flags
include(${PROJECT_UTILS_CMAKE_DIR}/default_flags.cmake)
### Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) # default to Release
endif()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread") # Needed for Topaze https://stackoverflow.com/a/61439452
### Additionnal build types
include(${PROJECT_UTILS_CMAKE_DIR}/additional_build_types.cmake)


# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
## System dependencies ###
if (NOT TARGET Python::Python OR NOT TARGET Python::NumPy)
  project_find_package(Python 3.8 REQUIRED COMPONENTS Development NumPy)
endif()
if (NOT TARGET MPI::MPI_CXX)
  project_find_package(MPI REQUIRED COMPONENTS CXX)
endif()
if (NOT TARGET Mpi4Py::Mpi4Py)
  project_find_package(Mpi4Py REQUIRED)
endif()

## Dependencies built from source ##
include(${PROJECT_UTILS_CMAKE_DIR}/find_or_fetch.cmake)
find_or_fetch_pybind11()


## By default, do not enable tests in dependencies if tests are not enabled in Maia
if (NOT ${PROJECT_NAME}_ENABLE_TESTS)
  option(std_e_ENABLE_TESTS "" OFF)
  option(cpp_cgns_ENABLE_TESTS "" OFF)
endif()

### ParaDiGM ###
if (NOT ${PROJECT_NAME}_BUILD_EMBEDDED_PDM)
  project_find_package(pdm CONFIG REQUIRED)
  project_find_package(pdma CONFIG)
else()
  # Maia uses ParaDiGM with these options
  set(PDM_ENABLE_SHARED          ON  CACHE BOOL "Maia uses ParaDiGM with shared libs"      FORCE)
  set(PDM_ENABLE_PYTHON_BINDINGS ON  CACHE BOOL "Maia uses ParaDiGM with python"           FORCE)
  set(PDM_ENABLE_MPI_CHECK       OFF CACHE BOOL "Maia uses ParaDiGM without MPI check"     FORCE)
  set(PASS_DEFAULT_FLAGS         ON  CACHE BOOL "Maia does not use ParaDiGM default flags" FORCE)
  # These option values are not FORCE, but the default is different than the one of ParaDiGM
  set(PDM_ENABLE_Fortran         OFF CACHE BOOL "Maia uses ParaDiGM without Fortran"    )
  set(PDM_ENABLE_STATIC          OFF CACHE BOOL "Maia uses ParaDiGM without static libs")
  set(PDM_ENABLE_UNIT_TEST       OFF CACHE BOOL "Maia uses ParaDiGM without unit tests" )
  set(PDM_ENABLE_TESTS           OFF CACHE BOOL "Maia uses ParaDiGM without func tests" )
  # Adding paradigm sources from external/paradigm
  include(${PROJECT_UTILS_CMAKE_DIR}/check_local_dependency.cmake)
  check_local_dependency(paradigm REQUIRED)
  project_add_subdirectory(paradigm)
endif()

### std_e + cpp_cgns ###
set(std_e_ENABLE_MPI ON CACHE BOOL "Maia uses std_e with mpi" FORCE)
project_add_subdir_or_package(std_e REQUIRED)
project_add_subdir_or_package(cpp_cgns REQUIRED)


# ------------------------------------------------------------------------------
# Compilation
# ------------------------------------------------------------------------------
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
    pdm::pdm_shared
    Mpi4Py::Mpi4Py
)

# ------------------------------------------------------------------------------
# Create file to source build environnement
# ------------------------------------------------------------------------------
include(${PROJECT_UTILS_CMAKE_DIR}/write_build_env_file.cmake)
write_build_env_file()

# ----------------------------------------------------------------------
# Copy logging.conf example to build folder
# ----------------------------------------------------------------------
if (EXISTS "${PROJECT_BINARY_DIR}/logging.conf")
  message("File ${PROJECT_BINARY_DIR}/logging.conf already present")
else()
  configure_file(
    ${PROJECT_SOURCE_DIR}/share/logging.conf.example
    ${PROJECT_BINARY_DIR}/logging.conf
    @ONLY
  )
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
target_install(${PROJECT_NAME})

# ------------------------------------------------------------------------------
# Python and wrapping
# ------------------------------------------------------------------------------
include(${PROJECT_UTILS_CMAKE_DIR}/python_wrapping.cmake)
compile_install_pybind_module(${PROJECT_NAME})
install_python_modules(${PROJECT_NAME})


# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
include(CTest)
if (${PROJECT_NAME}_ENABLE_TESTS)
  add_subdirectory(test)
endif()


# ------------------------------------------------------------------------------
# Build documentation
# ------------------------------------------------------------------------------
if(${PROJECT_NAME}_ENABLE_DOCUMENTATION)
  include(${PROJECT_UTILS_CMAKE_DIR}/build_documentation.cmake)
  build_sphinx_documentation()
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
get_directory_property(top_lvl_cmakelists PARENT_DIRECTORY)
if(top_lvl_cmakelists STREQUAL ${PROJECT_SOURCE_DIR})
  include(FeatureSummary)
  feature_summary(WHAT ALL)
endif()
