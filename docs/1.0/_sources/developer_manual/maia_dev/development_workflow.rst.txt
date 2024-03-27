Development workflow
====================

Sub-modules
-----------

The **Maia** repository is compatible with the development process described `here <https://github.com/BerengerBerthoul/project_utils/blob/master/doc/Git_workflow.md>`_. It uses git submodules to ease the joint development with other repositories compatible with this organization.

TL;DR: configure the git repository by sourcing `this file <https://github.com/BerengerBerthoul/project_utils/blob/master/git/submodule_utils.sh>`_ and then execute: 

.. code-block:: bash

  cd $MAIA_FOLDER
  git submodule update --init
  git_config_submodules


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
