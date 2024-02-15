Spack
=====


Installation through Spack
--------------------------

.. warning:: The procedure is not up-to-date

Maia depends on quite a few libraries of different kinds, be it system libraries like MPI, third-party libraries like HDF5, ONERA libraries like ParaDiGM , git submodules (std_e...), or Python modules (mpi4py, ruamel). The prefered way of installing Maia in a coherent environment is by using the `Spack package manager <https://spack.readthedocs.io/>`_. A Spack recipe for Maia can be found on the `ONERA Spack repository <https://gitlab.onera.net/informatics/infra/onera_spack_repo>`_.


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



Dependencies
^^^^^^^^^^^^

To get access to Maia dependencies in your development environment, you have several choices:

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

For development, you can still use Spack to have Maia dependencies, but then follow a typical CMake workflow with :code:`cmake/make`.
If you are developping within Maia and its submodules, you can filter them out from your Spack environment view like so:

.. code-block:: yaml

  view:
    default:
      exclude: ['maia','std-e','cpp-cgns','paradigm','pytest_parallel']

