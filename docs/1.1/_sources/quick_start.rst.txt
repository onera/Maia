.. _quick_start:

.. currentmodule:: maia

Quick start
===========

Environnements
--------------

Maia is now distributed in elsA releases (since v5.2.01) !

If you want to try the latest features, we provide ready-to-go environments including Maia and its dependencies on the following clusters:

**Spiro-EL8**

This is the recommended environment for standalone usage of Maia. It works with intel mpi library (2021)
and python version 3.9.

.. code-block:: sh

  source /scratchm/sonics/dist/source.sh --env maia
  module load maia/dev-default

If you want to use Maia within the standard Spiro environment, the next installation is compatible with
the socle socle-cfd/5.0-intel2120-impi:

.. code-block:: sh

  module use --append /scratchm/sonics/usr/modules/
  module load maia/dev-dsi-cfd5

Note that this is the environment used by elsA for its production spiro-el8_mpi.

**Sator**

Similarly, Maia installation are available in both the self maintained and standard socle
on Sator cluster. Sator's version is compiled with support of large integers.

.. code-block:: sh

  # Versions based on self compiled tools
  source /tmp_user/sator/sonics/dist/source.sh --env maia
  module load maia/dev-default

  # Versions based on socle-cfd compilers and tools
  module use --append /tmp_user/sator/sonics/usr/modules/
  module load maia/dev-dsi-cfd5


If you prefer to build your own version of Maia, see :ref:`installation` section.

Supported meshes
----------------

Maia supports CGNS meshes from version 4.2, meaning that polyhedral connectivities (NGON_n, NFACE_n
and MIXED nodes) must have the ``ElementStartOffset`` node.

Former meshes can be converted with the (sequential) maia_poly_old_to_new script included
in the ``$PATH`` once the environment is loaded:

.. code-block:: sh

  $> maia_poly_old_to_new mesh_file.hdf

The opposite maia_poly_new_to_old script can be used to put back meshes
in old conventions, insuring compatibility with legacy tools.

.. warning:: CGNS databases should respect the `SIDS <https://cgns.github.io/CGNS_docs_current/sids/index.html>`_.
  The most commonly observed non-compliant practices are:

  - Empty ``DataArray_t`` (of size 0) under ``FlowSolution_t`` containers.
  - 2D shaped (N1,N2) ``DataArray_t`` under ``BCData_t`` containers.
    These arrays should be flat (N1xN2,).
  - Implicit ``BCDataSet_t`` location for structured meshes: if ``GridLocation_t`` 
    and ``PointRange_t`` of a given ``BCDataSet_t`` differs from the
    parent ``BC_t`` node, theses nodes should be explicitly defined at ``BCDataSet_t``
    level.

  Several non-compliant practices can be detected with the ``cgnscheck`` utility. Do not hesitate
  to check your file if Maia is unable to read it.

Note also that ADF files are not supported; CGNS files should use the HDF binary format. ADF files can
be converted to HDF thanks to ``cgnsconvert``.

Highlights
----------

.. tip:: Download sample files of this section:
  :download:`S_twoblocks.cgns <../share/_generated/S_twoblocks.cgns>`,
  :download:`U_ATB_45.cgns <../share/_generated/U_ATB_45.cgns>`

.. rubric:: Daily user-friendly pre & post processing

Maia provides simple Python APIs to easily setup pre or post processing operations:
for example, converting a structured tree into an unstructured (NGon) tree
is as simple as

.. literalinclude:: snippets/test_quick_start.py
  :start-after: #basic_algo@start
  :end-before: #basic_algo@end
  :dedent: 2

.. image:: ./images/qs_basic.png
  :width: 75%
  :align: center

In addition of being parallel, the algorithms are as much as possible topologic, meaning that
they do not rely on a geometric tolerance. This also allow us to preserve the boundary groups
included in the input mesh (colored on the above picture).

.. rubric:: Building efficient workflows

By chaining this elementary blocks, you can build a **fully parallel** advanced workflow
running as a **single job** and **minimizing file usage**.

In the following example, we load an angular section of the
`ATB case  <https://turbmodels.larc.nasa.gov/axibump_val.html>`_,
duplicate it to a 180° case, split it, and perform some slices and extractions.

.. literalinclude:: snippets/test_quick_start.py
  :start-after: #workflow@start
  :end-before: #workflow@end
  :dedent: 2

.. image:: ./images/qs_workflow.png

The above illustration represents the input mesh (gray volumic block) and the
extracted surfacic tree (plane slice and extracted wall BC). Curved lines are
the outline of the volumic mesh after duplication.

.. rubric:: Compliant with the pyCGNS world

Finally, since Maia uses the standard `CGNS/Python mapping 
<https://cgns.github.io/CGNS_docs_current/python/sidstopython.pdf>`_,
you can set up applications involving multiple python packages:
here, we create and split a mesh with maia, but we then call Cassiopee functions
to compute the gradient of a field on each partition.

.. literalinclude:: snippets/test_quick_start.py
  :start-after: #pycgns@start
  :end-before: #pycgns@end
  :dedent: 2

.. image:: ./images/qs_pycgns.png

Be aware that other tools often expect to receive geometrically
consistent data, which is why we send the partitioned tree to
Cassiopee functions. The outline of this partitions (using 4 processes)
are materialized by the white lines on the above figure.


Resources and Troubleshouting
-----------------------------

The :ref:`user manual <user_manual>` describes most of the high level APIs provided by Maia.
If you want to be comfortable with the underlying concepts of distributed and
partitioned trees, have a look at the :ref:`introduction <intro>` section.

The user manual is illustrated with basic examples. Additional
test cases can be found in 
`the sources <https://gitlab.onera.net/numerics/mesh/maia/-/tree/dev/test>`_.

Issues can be reported on 
`the gitlab board <https://gitlab.onera.net/numerics/mesh/maia/-/issues>`_
and help can also be asked on the dedicated
`Element room <https://synapse.onera.fr/#/room/!OcgyjrTUlihJWTYWbq:synapse.onera.fr?via=synapse.onera.fr>`_.
