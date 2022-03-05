.. _cgns_trees:

Maia CGNS trees
===============

Maia algorithms extensively use Python/CGNS trees (**pytrees**). However, most of the times, functions don't accept any CGNS tree: they may use a **distributed tree**, a **partitioned tree** or trees with some additional properties. These trees are described here.

Distributed and partitioned trees
---------------------------------

Let us use the following tree as an example:

.. image:: ./images/trees/tree_seq.png

This tree is a **global tree**. It may appear like that on a HDF5/CGNS file, or if loaded entirely on one process as a Python/CGNS tree.

.. _dist_tree:

Distributed trees
^^^^^^^^^^^^^^^^^

A :def:`dist tree` is a CGNS tree where the tree structure is replicated across all processes, but array values of the nodes are distributed, that is, each process only stores a block of the complete array.

If we distribute our tree over two processes, we would then have something like that:

.. image:: ./images/trees/dist_tree.png

Let us look at one of them and annotate nodes specific to the distributed tree:

.. image:: ./images/trees/dist_tree_expl.png

Arrays of non-constant size are distributed: fields, connectivities, :cgns:`PointLists`.
Others (:cgns:`PointRanges`, :cgns:`CGNSBase_t` and :cgns:`Zone_t` dimensions...) are of limited size and therefore replicated on all processes with virtually no memory penalty.

On each process, for each entity kind, a **partial distribution** is stored, that gives information of which block of the arrays are stored locally.

For example, for process 0, the distribution array of vertices of :cgns:`MyZone` is located at :cgns:`MyBase/MyZone/Distribution/Vertex` and is equal to :code:`[0, 9, 18]`. It means that only indices in the semi-open interval :code:`[0 9)` are stored by the **dist tree** on this process, and that the total size of the array is :code:`18`.
This partial distribution applies to arrays spaning all the vertices of the zone, e.g. :cgns:`CoordinateX`.

More formally, a :def:`partial distribution` related to an entity kind :code:`E` is an array :code:`[start,end,total_size]` of 3 int64 where :code:`[start:end)` is a closed/open interval giving, for all global arrays related to :code:`E`, the sub-array that is stored locally on the distributed tree, and :code:`total_size` is the global size of the arrays related to :code:`E`.

The distributed entities are:

.. glossary::
      Vertices and Cells
        The **partial distribution** are stored in :cgns:`Distribution/Vertex` and :cgns:`Distribution/Cell` nodes at the level of the :cgns:`Zone_t` node.

        Used for example by :cgns:`GridCoordinates_t` and :cgns:`FlowSolution_t` nodes if they do not have a :cgns:`PointList` (i.e. if they span the entire vertices/cells of the zone)

      Quantities described by a :cgns:`PointList` or :cgns:`PointRange`
        The **partial distribution** is stored in a :cgns:`Distribution/Index` node at the level of the :cgns:`PointList/PointRange`

        For example, :cgns:`ZoneSubRegion_t` and :cgns:`BCDataSet_t` nodes.

        If the quantity is described by a :cgns:`PointList`, then the :cgns:`PointList` itself is distributed the same way (in contrast, a :cgns:`PointRange` is fully replicated across processes because it is lightweight)

      Connectivities
        The **partial distribution** is stored in a :cgns:`Distribution/Element` node at the level of the :cgns:`Element_t` node. Its values are related to the elements, not the vertices of the connectivity array.

        If the element type is heterogenous (NGon, NFace or MIXED) a :cgns:`Distribution/ElementConnectivity` is also present, and this partial distribution is related to the :cgns:`ElementConnectivity` array.

.. _part_tree:

Partitioned trees
^^^^^^^^^^^^^^^^^

A :def:`part tree` is a partial CGNS tree, i.e. a tree for which each zone is only stored by one process. Each zone is fully stored by its process.

If we take the global tree from before and partition it, we may get the following tree:

.. image:: ./images/trees/part_tree.png

If we annotate the first one:

.. image:: ./images/trees/part_tree_expl.png

A **part tree** is just a regular tree with additional information (in the form of :cgns:`GlobalNumbering` nodes) that keeps the link with the unpartitioned tree it comes from. Notice that the tree structure is **not** the same across all processes.

The :cgns:`GlobalNumbering` nodes are at exactly the same positions that the :cgns:`Distribution` nodes were in the distributed tree.

A :cgns:`GlobalNumbering` contains information to link an entity in the partition to its corresponding entity in the original tree. For example, the element section :cgns:`Hexa` has a global numbering array of value :code:`[3 4]`. It means:

* Since it is an array of size 2, there is 2 elements in this section (which is confirmed by the :cgns:`ElementRange`) ,
* The first element was the element of id :code:`3` in the original mesh,
* The second element was element :code:`4` in the original mesh.

Naming conventions
""""""""""""""""""

When partitioning, some nodes are split, so there is convention to keep track of the fact they come from the same original node:

* :cgns:`Zone_t` nodes : :cgns:`MyZone` is split in :cgns:`MyZone.PX.NY` where `X` is the rank of the process, and `Y` is the id of the zone on process `X`.
* Splitable nodes (notably :cgns:`GC_t`) : :cgns:`MyNode` is split in :cgns:`MyNode.N`. They appear in the following scenario:

  * We partition for 3 processes
  * :cgns:`Zone0` is connected to :cgns:`Zone1` through :cgns:`GridConnectivity_0_to_1`
  * :cgns:`Zone0` is not split (but goes to process 0 and becomes :cgns:`Zone0.P0.N0`). Zone1 is split into :cgns:`Zone1.P1.N0` and :cgns:`Zone1.P2.N0`. Then :cgns:`GridConnectivity_0_to_1` of :cgns:`Zone0` must be split into :cgns:`GridConnectivity_0_to_1.1` and :cgns:`GridConnectivity_0_to_1.2`.

Note that partitioning may induce new :cgns:`GC_t` internal to the original zone being splitted. Their name is implementation-defined and those nodes do not have a :cgns:`GlobalNumbering` since they did not exist in the original mesh.

.. _maia_tree:

Maia trees
----------

A CGNS tree is said to be a :def:`Maia tree` if it has the following properties:

* For each unstructured zone, the :cgns:`ElementRange` of all :cgns:`Elements_t` sections

  * are contiguous
  * are ordered by ascending dimensions (i.e. edges come first, then faces, then cells)
  * the first section starts at 1
  * there is at most one section by element type (e.g. not possible to have two :cgns:`QUAD_4` sections)

Notice that this is property is required by **some** functions of Maia, not all of them!

A **Maia tree** may be a **global tree**, a **distributed tree** or a **partitioned tree**.
