.. _installation:

Installation
############

Prefered installation procedure
===============================

Maia depends on quite a few libraries of different kinds, be it system libraries like MPI, third-party libraries like HDF5, ONERA libraries like ParaDiGM and Cassiopée, git submodules (std_e...), or Python modules (mpi4py, ruamel). The prefered way of installing Maia in a coherent environment is by using the `Spack package manager <https://spack.readthedocs.io/>`_. A Spack recipe for Maia can be found on the `ONERA Spack repository <https://gitlab.onera.net/informatics/infra/onera_spack_repo>`_.

Installation through Spack
--------------------------

1. Source a Spack repository on your machine.
2. If you don't have a Spack repository ready, you can download one with :code:`git clone https://github.com/spack/spack.git`. On ONERA machines, it is advised to use the `Spacky <https://gitlab.onera.net/informatics/infra/spacky>`_ helper.
3. Download the **ONERA Spack repository** with :code:`git clone https://gitlab.onera.net/informatics/infra/onera_spack_repo.git`
4. Tell Spack that package recipes are in :code:`onera_spack_repo` by adding the following lines to :code:`$SPACK_ROOT/etc/repos.yaml`:

.. code-block:: yaml

  repos:
  - path/to/onera_spack_repo

(note that **spacky** does steps 3. and 4. for you)

5. You should be able to see the package options of Maia with :code:`spack info maia`
6. To install Maia: :code:`spack install maia`


Development workflow
--------------------

For development, it is advised to use Spack to have Maia dependencies, but then follow a typical CMake workflow with :code:`cmake/make`.


Dependencies
^^^^^^^^^^^^

To get access to Maia dependencies in your development environment, you can:

* Install a Spack version of Maia, source it in your development environment to get all the dependencies, then override with your own compiled version of Maia
* Do the same, but use a Spack environment containing Maia instead of just the Maia package
* Source a Spack environment where Maia has been removed from the environment view. This can be done by adding the following lines to the :code:`spack.yaml` environement file:

.. code-block:: yaml

  view:
    default:
      exclude: ['maia']

This last option is cleaner because you are sure that you are not using another version of Maia (but it means you need to create or have access to such an environment view)

Source the build folder
^^^^^^^^^^^^^^^^^^^^^^^

You can develop without the need to install Maia. However, in addition to sourcing your dependencies in your development environment, you also need to source the build artifacts by:

.. code-block:: bash

  cd $MAIA_BUILD_FOLDER
  source source.sh

The :code:`source.sh` file is created by CMake and will source all Maia artifacts (dynamic libraries, python modules...)


Development workflow with submodules
------------------------------------

It is often practical to develop Maia with some of its dependencies, namely:

* project_utils
* std_e
* cpp_cgns
* paradigm
* pytest_parallel

For that, you need to use git submodules. Maia submodules are located at :code:`$MAIA_FOLDER/external`. To populate them, use :code:`git submodule update --init`. Once done, CMake will use these versions of the dependencies. If you don't populate the submodules, CMake will try to use the ones of your environment (for instance, the one installed by Spack).

We advise that you use some additional submodule configuration utilities provided in `this file <https://github.com/BerengerBerthoul/project_utils/blob/master/git/submodule_utils.sh>`_. In particular, you should use:

.. code-block:: bash

  cd $MAIA_FOLDER
  git submodule update --init
  git_config_submodules

The detailed meaning of `git_config_submodules` and the git submodule developper workflow of Maia is presented `here <https://github.com/BerengerBerthoul/project_utils/blob/master/doc/Git_workflow.md>`_.

If you are using Maia submodules, you can filter them out from your Spack environment view like so:

.. code-block:: yaml

  view:
    default:
      exclude: ['maia','std-e','cpp-cgns','paradigm','pytest_parallel']

Manual installation procedure
=============================

Dependencies
------------

**Maia** depends on:

* python3
* MPI
* hdf5

* Cassiopée

* pytest >6 (python package)
* ruamel (python package)
* mpi4py (python package)

The build process requires:

* Cmake >= 3.14
* GCC >= 8 (Clang and Intel should work but no CI)


Other dependencies
^^^^^^^^^^^^^^^^^^

During the build process, several other libraries will be downloaded:

* pybind11
* range-v3
* doctest

* ParaDiGM
* project_utils
* std_e
* cpp_cgns

This process should be transparent.


Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

The documentation build requires:

* Doxygen >= 1.8.19
* Breathe >= 4.15 (python package)
* Sphinx >= 3.00 (python package)

Build and install
-----------------

1. Install the required dependencies. They must be in your environment (:code:`PATH`, :code:`LD_LIBRARY_PATH`, :code:`PYTHONPATH`).

 For pytest, you may need these lines :

.. code:: bash

  pip3 install --user pytest
  pip3 install --user pytest-mpi
  pip3 install --user pytest-html
  pip3 install --user pytest_check
  pip3 install --user ruamel.yaml

2. Then you need to populate your :code:`external` folder. You can do it with :code:`git submodule update --init`

3. Then use CMake to build maia, e.g.

.. code:: bash

  SRC_DIR=<path to source repo>
  BUILD_DIR=<path to tmp build dir>
  INSTALL_DIR=<path to where you want to install Maia>
  cmake -S $SRC_DIR -B$BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cd $BUILD_DIR && make -j && make install

