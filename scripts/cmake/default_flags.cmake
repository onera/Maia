#------------------------------------------------------------------------------
# Default flags for several compilers
#------------------------------------------------------------------------------

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

if (NOT MAIA_SKIP_DEFAULT_FLAGS)
  if (CMAKE_CXX_FLAGS)
    message(WARNING "Your C++ flags will be overriden. Use MAIA_SKIP_DEFAULT_FLAGS if you want to use them")
  endif()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fmax-errors=4 ")
    set (CMAKE_CXX_FLAGS_SANITIZE "-O0 -g -fsanitize=address -fno-omit-frame-pointer -Wall -Wextra ")
    # Other interesting flags
    #-ftree-vectorize -ftree-loop-vectorize -fvect-cost-model=unlimited -mprefer-vector-width=512
    #-ftree-loop-optimize -ftree-vectorize -fopt-info -fopt-info-all
    #-fprofile-arcs -ftest-coverage --coverage

  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set (CMAKE_CXX_FLAGS "-Wall -Wcheck -Wpointer-arith")
    set (CMAKE_CXX_FLAGS_SANITIZE "-g -O0 -traceback -w2")
    #set (CXX_LIBRARIES -cxxlib)

  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpointer-arith -Wno-missing-braces") # missing-braces disabled, else GCC warning
    set (CMAKE_CXX_FLAGS_SANITIZE "-O0 -g -fsanitize=address -fno-omit-frame-pointer")

  else()
    message (WARNING "Default flags are not defined for ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set (CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_RELEASE} -march=native -p" CACHE STRING "Flags used for profiling." FORCE)
  set (CMAKE_CXX_FLAGS_SANITIZE  "${CMAKE_CXX_FLAGS_SANITIZE}" CACHE STRING "Flags used by the compiler during sanitize builds" FORCE)

  mark_as_advanced (CMAKE_CXX_FLAGS_PROFILING CMAKE_CXX_FLAGS_SANITIZE MAIA_SKIP_DEFAULT_FLAGS)

  set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profiling Sanitize."
      FORCE)

endif()
