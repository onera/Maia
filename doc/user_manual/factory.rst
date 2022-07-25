Factory module
==============

The ``maia.factory`` module can be used to generate parallel CGNS trees. Those can
be obtained from an other kind of tree, or can be generated from user parameters.

Generation
----------

.. autofunction:: maia.factory.generate_dist_block

Partitioning
------------

.. autofunction:: maia.factory.partition_dist_tree

Partitioning options
^^^^^^^^^^^^^^^^^^^^
Partitioning can be customized with the following keywords arguments:

.. py:attribute:: graph_part_tool

    Graph partitioning library to use to split unstructured blocks. Irrelevent for structured blocks.

    :Admissible values: ``parmetis``, ``ptscotch``, ``hilbert``. Blocks defined by nodal elements does 
      not support hilbert method.
    :Default value: ``parmetis``, if installed; else ``ptscotch``, if installed; ``hilbert`` otherwise.

.. py:attribute:: zone_to_parts

    Control the number, size and repartition of partitions. See :ref:`user_man_part_repartition`.

    :Default value: Computed such that partitioning is balanced using
      :func:`maia.factory.partitioning.compute_balanced_weights`.

.. py:attribute:: part_interface_loc

    :cgns:`GridLocation` for the created partitions interfaces. Pre-existing interface keep their original location.

    :Admissible values: ``FaceCenter``, ``Vertex``
    :Default value: ``FaceCenter`` for unstructured zones with NGon connectivities; ``Vertex`` otherwise.

.. py:attribute:: reordering

    Dictionnary of reordering options, which are used to renumber the entities in each partitioned zone. See
    corresponding documentation.

.. py:attribute:: preserve_orientation

    If True, the created interface faces are not reversed and keep their original orientation. Consequently,
    NGonElements can have a zero left parent and a non zero right parent.
    Only relevant for U/NGon partitions.

    :Default value: ``False``

.. py:attribute:: dump_pdm_output

    If True, dump the raw arrays created by paradigm in a :cgns:`CGNSNode` at (partitioned) zone level. For debug only.

    :Default value: ``False``

.. _user_man_part_repartition:

Repartition         
^^^^^^^^^^^

The number, size, and repartition (over the processes) of the created partitions is
controlled through the ``zone_to_parts`` keyword argument: each process must provide a
dictionary associating the path of every distributed zone to a list of floats.
For a given distributed zone, current process will receive as many partitions
as the length of the list (missing path or empty list means no partitions); the size of these partitions
corresponds to the values in the list (expressed in fraction of the input zone).
For a given distributed zone, the sum of all the fractions across all the processes must
be 1.

This dictionary can be created by hand; for convenience, Maia provides two functions in the
:mod:`maia.factory.partitioning` module to create this dictionary.

.. autofunction:: maia.factory.partitioning.compute_regular_weights
.. autofunction:: maia.factory.partitioning.compute_balanced_weights

Reordering options
^^^^^^^^^^^^^^^^^^

For unstructured zones, the reordering options are transmitted to ParaDiGM in order to 
control the renumbering of mesh entities in the partitions.

+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| Kwarg              | Admissible values                 | Effect                              | Default                    |
+====================+===================================+=====================================+============================+
| cell_renum_method  | "NONE", "RANDOM", "HILBERT",      | Renumbering method for the cells    | NONE                       |
|                    | "CUTHILL", "CACHEBLOCKING",       |                                     |                            |
|                    | "CACHEBLOCKING2", "HPC"           |                                     |                            |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| face_renum_method  | "NONE", "RANDOM", "LEXICOGRAPHIC" | Renumbering method for the faces    | NONE                       |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| vtx_renum_method   | "NONE", "SORT_INT_EXT"            | Renumbering method for the vertices | NONE                       |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| n_cell_per_cache   | Integer >= 0                      | Specific to cacheblocking           | 0                          |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| n_face_per_pack    | Integer >= 0                      | Specific to cacheblocking           | 0                          |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+
| graph_part_tool    | "parmetis", "ptscotch"            | Graph partitioning library to       | Same as partitioning tool  |
|                    |                                   | use for renumbering                 |                            |
+--------------------+-----------------------------------+-------------------------------------+----------------------------+

Recovering from partitions
--------------------------

.. autofunction:: maia.factory.recover_dist_tree
