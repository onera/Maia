.. contents:: :local:

.. _log:

Introduction
============

Vocabulary
----------

Global data
^^^^^^^^^^^

**Global data** is the complete data that describes an object.


Distributed data
^^^^^^^^^^^^^^^^

**Distributed data** is data that is stored over multiple memory spaces. The data can't be accessed completely by one process. It has to be distributed over memory either because it is too heavy, or in order to take advantage of parallel algorithms.

A **block** of distributed data is the portion of that data that is stored over one memory space. Each block can only be interpreted as a piece of the *global data*.

Example: a field array where portions of the data are stored on multiple computer nodes.

Partitions
^^^^^^^^^^

Contrary to a *block* of distributed data, a **partition** is a coherent data structure that can be operated in semi-isolation. In order to take advantage of parallel algorithms, most of the time we want to deal with multiple partitions on multiple memory spaces. Of course, partitions are linked together, but the idea is to alternate between isolated computations on each partition, and information exchange between partitions.

Example : a CGNS zone.

Collective data and operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Collective data** over a set of processes is a piece of the same data that is repeated on the memory space of each process.

Example : a **distribution array** (see below).

A **collective operation** is an action in which a set of processes must participate in.

Example: a broadcast.


Distribution
^^^^^^^^^^^^

A **distribution array**, or **distribution**, is an array which describes how *global data* is *distributed* over memory spaces. Say that I have a global array of 500 elements and it is uniformly distributed over 5 processes. Then is distribution would be the array :code:`dist=[0,100,200,300,400,500]`. Process :code:`i` will store a *block* of data spanning the semi-open interval :code:`[dist[i],dist[i+i])`.

Distribution arrays are most of the time *collective data*, because each process holding a *block* of data needs to know which range of the data it is holding, and which range the others are holding.


Local and global numbering
^^^^^^^^^^^^^^^^^^^^^^^^^^

If global data were to be seen only as distributed blocks of memory over processes, life would be (relatively) simple. However, many algorithms
require to operate on partitions. Hence, global data has to be partitioned.

However, the link between an entity in a partition (say, a vertex in a mesh partition) and the entity of the global data it was created from (the same vertex, but in the original, global mesh) must be kept for multiple reasons. Maybe the most important one is that during the partitioning process, some entities of the global data are duplicated over multiple partitions (e.g. the matching vertices of two partitions), but they still represent the same data (they represent the same original vertex).

In order to know, for an element of a partition, which global entity it represents, we use a **local to global numbering array** (often called :code:`LN_to_GN` for short). Each partition has a :code:`LN_to_GN` array. For an element at index :code:`i` in array :code:`A` (called the **local numbering**), :code:`LN_to_GN[i]` gives the **global numbering**, that is, the global identifier of the element in the *global* array.


TODO prez de référence


Parallel CGNS trees
^^^^^^^^^^^^^^^^^^^

Depending on the purpose, we need to use multiple type of CGNS trees.

A **full tree** is a tree as it is inside a CGNS file, or how it would be loaded by only one process. A full tree is *global data*.

A **dist tree** is a distributed tree, i.e. for each node of the tree, each process only stores a block of the complete array value. See :ref:`load_dist_tree`.

A **part tree** is a partial tree, i.e. a tree for which each zone is only stored by one process. Each zone is fully stored by its process.

A **size tree** is a tree in which only the size of the data is stored. A *size tree* is typically *collective* because each process needs it to know which *block* of data it will have to load and store.

(A **skeleton tree** is a collective tree in which fields and element connectivities are not loaded)

Typical workflow
----------------

Most of the time, the mesh we want to operate on is not partitioned. This is mainly due to the fact that the partitoning we want depends on the number of processes we want to use, and this number varies. The typical workflow one wants to use is the following:

1. Begin with a non-partitionned tree. The tree may have several zones because of the configuration of the mesh (e.g. multiple stages in turbomachinery), but the zone are not *a priori* the ones that we want for our CFD computation (e.g. because the zones are too big, or unbalanced).
2. Load this tree as a *dist tree*. See :ref:`load_dist_tree`
3. A *part tree* is computed from the *dist tree* by calling graph partitioning algorithms, then transfering fields. The *part tree* contains :code:`LN_to_GN` information to keep the link with the *dist tree* it has been generated from.
4. The solver is called over the *part tree*
5. The result fields are transfered back to the *dist tree*
6. The updated *dist tree* is saved to disk.

Other workflows and refinements
-------------------------------

Merging partitions
^^^^^^^^^^^^^^^^^^

Since partitioning depends on the number of ressources we want to use, it is a computation strategy detail and it should not be kept when saving a file. As a matter of fact, inside the global mesh, the one saved to disk, zones should only materialize different components (e.g. multiple stages in turbumachinery), NOT different partitions.

If this is not the case, we may want to merge zones. Indeed, it may simplify pre/post-processing of the mesh. Plus, the bigger the zone is, the more freedom there will be to optimize partitioning.

Note: As long a the :code:`LN_to_GN` arrays are kept, merging partitions back to the original mesh is easy.

Generating element connectivities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO:

vocabulary cell_txt....

* cell_vtx -> face_vtx, face_cell ("fetch")
* face_vtx, face_cell -> cell_vtx
* cell_cell, vtx_cell, vtx_vtx, edge_vtx

Grid connectivities
^^^^^^^^^^^^^^^^^^^

* face -> vtx
* vtx -> face

Renumbering
^^^^^^^^^^^

* partitions alone
* partitions + update LN_to_GN


Extended partitions
^^^^^^^^^^^^^^^^^^^

* Ghost cells, ghost nodes
* Reveral ranks


