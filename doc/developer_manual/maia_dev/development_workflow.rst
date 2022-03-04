Development workflow
====================

Sub-modules
-----------

The **Maia** repository is compatible with the development process described in `external/project_utils/doc/main.md`. It uses git submodules to ease the joint development with other repositories compatible with this organization. TL;DR: configure the git repository with `cd external/project_utils/scripts && configure_top_level_repo.sh`.


Launch tests
------------

Tests can be launched by calling CTest, but during the development, we often want to parameterize how to run tests (which ones, number of processes, verbosity level...).

There is a :code:`source.sh` generated in the :code:`build/` folder. It can be sourced in order to get the correct environment to launch the tests (notably, it updates :code:`LD_LIBRARY_PATH` and :code:`PYTHONPATH` to point to build artifacts).

Tests can be called with e.g.:

.. code:: bash

  cd $PROJECT_BUILD_DIR
  source source.sh
  mpirun -np 4 external/std_e/std_e_unit_tests
  ./external/cpp_cgns/cpp_cgns_unit_tests
  mpirun -np 4 test/maia_doctest_unit_tests
  mpirun -np 4 pytest $PROJECT_SRC_DIR/maia --with-mpi
