.. _quick_start:

.. currentmodule:: maia

Quick start
===========

.. contents:: :local:

Introduction
------------

**Maia** offers parallel algorithms over CGNS trees: distributed loading, partitioning, and various kind of transformations (generation of faces, generation of ghost cells...).


Installation
------------

See :ref:`installation` to build Maia, execute the tests, build the documentation and install the library.

Highlights
----------

Distributed trees algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maia main algorithms operate on distributed CGNS trees. A distributed tree over an MPI communicator is like a regular CGNS tree where all ranks have the same tree structure, but each rank only holds a portion of the arrays. Distributed trees algorithms are always collective on their associated MPI communicator. A detailed description of distributed trees is given in :ref:`this section <dist_tree>`.

Sample of the main distributed tree algorithms:

.. code-block:: python

  import maia
  from mpi4py import MPI
  comm = MPI.COMM_WORLD

  # Load a distributed CGNS tree from a file
  dist_tree = maia.io.file_to_dist_tree('file.cgns', comm)

  # Write a distributed CGNS tree
  maia.io.dist_tree_to_file(dist_tree, 'file.hdf', comm)

  # Generate a distributed cube
  dist_tree = maia.generate.dcube_generator(n_vtx, lenght, origin, comm)

  # Structured to unstructured
  dist_tree_u = maia.transform.convert_s_to_u(dist_tree_s, comm)

  # Transform a tree with standard element types to one with NGon/NFace sections
  maia.elements_to_ngons(dist_tree_elts, comm)


Partitioned/Distributed trees algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While distributed trees are useful for many pre/post-treatment algorithms, they are not fit for other algorithms. In particular, solver algorithms operate on partitioned trees. A partitioned tree is a CGNS tree that results from the partition of a base tree: each zone of the base tree has been split into sub-zones that are distributed over the ranks of a communicator. While an algorithm may operate on the sole partitioned tree, we generally need to keep its relationship with its base tree. Since the base tree is of arbitrary size, it is generally treated by Maia as a distributed tree.

Sample of the main partitioned tree algorithms:

.. code-block:: python

   # Partition a distributed tree
   part_tree = maia.partitioning(dist_tree, comm)

   # Generate a dist_tree from a part_tree
   dist_tree = maia.disttree_from_parttree(part_tree, comm)

   # Transfer fields from a zone of a dist_tree to a part_tree
   maia.dist_sol_to_part_sol(dist_zone, part_zones, comm)

   # Transfer fields from a zone of a part_tree to a dist_tree
   maia.part_sol_to_dist_sol(dist_zone, part_zones, comm)
