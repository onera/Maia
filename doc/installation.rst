.. _installation:

Installation
############

Dependencies
------------

**Maia** depends on:

* :code:`Python` >= 3.8
* :code:`MPI`

* :code:`mpi4py` (python package)
* :code:`h5py` with :code:`MPI` (python package)

* :code:`ParMetis` (optional, for partitioning)
* :code:`PtScotch` (optional, for partitioning)

The build process requires:

* :code:`CMake` >= 3.14
* :code:`GCC` >= 8 (:code:`clang` and Intel :code:`icpx` should work but are not tested by CI)
* :code:`PyBind11` >= 2.8.1
* :code:`Cython` 0.29 (needed by ParaDiGM, :ref:`see below <pdm_install>`)

.. note:: For convenience, CMake will automatically download PyBind11 from GitHub if it does not find it in your environment.

.. warning:: The :code:`h5py` package must use the **MPI version of HDF5**, and both :code:`h5py` and :code:`mpi4py` must use the **same MPI** library. Make sure to use a coherent environment with parallel versions of the libraries!

.. warning:: :code:`Cython` >= 3 is **not supported** for now. You have to use the legacy branch.


Build from source with CMake
----------------------------

First, get the sources of Maia with Git, and retrieve the sources of its submodules (they will be downloaded from GitHub):

.. code:: bash

  git clone git@gitlab.onera.net:numerics/mesh/maia.git
  cd maia
  git submodule update --init


If you have access to the restricted ParaDiGMA algorithms, you may want to use Maia with them. For that:

.. code:: bash

  (cd external/paradigm && git submodule update --init)

Then you can configure and build using CMake:

.. code:: bash

  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=<your/installation/path>
  make -j
  make install


Here are some useful CMake flags:

* If you want to use 32-bit integers global indexing, use :code:`PDM_ENABLE_LONG_G_NUM=OFF`.
* If you want to use ParaDiGMA features, use :code:`PDM_ENABLE_EXTENSION_PDMA=ON`.
* If your compiler is not C++20-compliant (most notably GCC 8), use :code:`CMAKE_CXX_STANDARD=17`. Some functionnalities will be missing.

.. _pdm_install:

* If you want to use an installation of ParaDiGM already present in your environment, use :code:`maia_BUILD_EMBEDDED_PDM=OFF`.
  For that, you need the versions of ParaDiGM and Maia to be compatible:

  +--------------+--------+-------------+--------+-------------+-------------+
  | **Maia**     | v1.0   | v1.1 & v1.2 | v1.3   | v1.4 & v1.5 | v1.6 & v1.7 |
  +--------------+--------+-------------+--------+-------------+-------------+
  | **ParaDiGM** | v2.2.x | v2.3.x      | v2.4.x | v2.5.x      | v2.6.x      |
  +--------------+--------+-------------+--------+-------------+-------------+

  If you are using a development version of Maia, then you can't use :code:`maia_BUILD_EMBEDDED_PDM=OFF`.


Documentation and tests
-----------------------

.. rubric:: Tests

Running Maia tests requires:

* :code:`doctest` (C++ library)
* :code:`pytest` > 6 (python package)
* :code:`pyyaml` (python package)
* :code:`h5ls` and :code:`h5diff` (HDF5 utilities that generally come with the HDF5 library)

If :code:`doctest` is not found on the environment, it will be downloaded from GitHub by CMake. This should be transparent.

Tests are built by default. You can turn them off with :code:`maia_ENABLE_TESTS=OFF`.


.. rubric:: Documentation

The documentation build requires:

* :code:`Sphinx` >= 3.00 (python package)

Configure CMake with :code:`maia_ENABLE_DOCUMENTATION=ON` to enable the documentation. Then generate it with :code:`make maia_sphinx`.


