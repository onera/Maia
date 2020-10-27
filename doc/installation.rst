.. _installation:

Installation
============

Dependencies
------------

**Maia** depends on :
* python
* cython
* pytest

* mpi
* ParaDiGM

* hdf5

* Cassiopée


The build process requires:

* Cmake >= 3.12
* GCC >= 8 (Clang and Intel should work but no CI)


Other dependencies
^^^^^^^^^^^^^^^^^^
During the build process, several other libraries will be downloaded:

* pybind11
* range-v3
* doctest

* project_utils
* std_e
* cpp_cgns

The process should be transparent to the user.

TODO: cython, pytest, paradigm, Cassiopée should be here

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
The documentation build requires:

* Doxygen >= 1.8.19
* Breathe >= 4.15 (python package)
* Sphinx >= 3.00 (python package)

Build and install
-----------------

1. Install the required dependencies. They must be in your environment (`PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`).

 For pytest, you may need these lines :

.. code:: bash

  pip3 install --user pytest-mpi
  pip3 install --user pytest-html
  pip3 install --user pytest_check

2. Then you need to populate your :code:`external` folder. If you got Maia from a `Maia_suite` repository, then there is nothing to do. Else, you can do it with `git submodule update --init`

3. Then use CMake to build maia, e.g. 

.. code:: bash
  SRC_DIR=<path to source repo>
  BUILD_DIR=<path to tmp build dir>
  INSTALL_DIR=<path to where you want to install Maia>
  cmake -S $SRC_DIR -B$BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cd $BUILD_DIR && make -j16 && make install

CMake options
^^^^^^^^^^^^^

* MAIA_PDM_IN_EXTERNAL_FOLDER: If ON, then ParaDiGM will be built from the `maia/exteral/paradigm` folder. Else, CMake will try to find ParaDiGM in your environment. ON by default.
* MAIA_ENABLE_MPI_TESTS: If ON, CTest will try to execute the parallel test suite. OFF by default.
