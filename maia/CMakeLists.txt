# --------------------------------------------------------------
set(PYMODULES_NAME # cgns_registry
                   # connectivity
                   # geometry
                   # io
                   # mesh_gen
                   # partitioning
                   # tree_exchange
                   utils
)
# --------------------------------------------------------------


# --------------------------------------------------------------
# set(CORE_PYMODULES)
# # --------------------------------------------------------------

# # --------------------------------------------------------------
foreach(pymod IN LISTS PYMODULES_NAME)
  # message(${SHN}_${pymod})
  # list(APPEND CORE_PYMODULES "${pymod}")

  add_subdirectory(${pymod})

  # install(DIRECTORY      "${CMAKE_SOURCE_DIR}/${pymod}" # source directory
  #         DESTINATION    "include"                      # target directory
  #         FILES_MATCHING                                # install only matched files
  #         PATTERN        "*.hpp"                        # select header files
  #         PATTERN        "*.h"
  #         )
endforeach ()
# --------------------------------------------------------------

mixpython_cython_add_module(maia)

# # --------------------------------------------------------------
# set(LIBRARIES_CORE_NAME globgraph_core)
# # --------------------------------------------------------------

# # --------------------------------------------------------------
# set(CORE_LIBRARIES)
# # --------------------------------------------------------------
