.. _user_algo:

Algo module
===========

The ``maia.algo`` module provides various algorithms to be applied to one of the
two kind of trees defined by Maia:

- ``maia.algo.dist`` module contains some operations applying on distributed trees
- ``maia.algo.part`` module contains some operations applying on partitioned trees

In addition, some algorithms can be applied indistinctly to distributed or partitioned trees.
These algorithms are accessible through the ``maia.algo`` module.

The ``maia.algo.seq`` module contains a few sequential utility algorithms.


.. rubric:: Generalities

Here are some remarks applying to all the functions:

- Unless something else specified, functions operate inplace (input tree is modified) and returns ``None``.
- The ``comm`` argument always refers to the MPI communicator used to
  create the input tree.
- Argument ``containers_name`` is used by some functions transfering data fields
  (eg :func:`~maia.algo.interpolate`). Such function operates on a 'per-container' basis,
  meaning that only the requested containers (FlowSolution_t, DiscreteData_t or
  ZoneSubRegion_t nodes) will be treated.
  Depending on the function, supported containers can be either

  - **full**: data exists for all points or elements of all input zones (typically a FlowSolution);
  - **partial**: data exists for a susbet of points or elements on some input zones (typically a ZoneSubRegion).

  Expected value for ``containers_name`` is a list of ``str`` or the shortcut ``'ALL'``,
  in which case the fonction selects all the admissible containers.
- In this documentation, the term *join* (shortcut: *jn*) is used to designate matching interfaces, *ie*
  ``GridConnectivity1to1_t`` nodes and ``GridConnectivity_t`` nodes of
  :func:`~maia.pytree.GridConnectivity.Type` ``Abutting1to1``.


.. _user_man_dist_algo:

Distributed algorithms
----------------------

The following algorithms apply on maia distributed trees.


Connectivities conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.convert_s_to_u
.. autofunction:: maia.algo.dist.convert_elements_to_ngon
.. autofunction:: maia.algo.dist.convert_ngon_to_elements
.. autofunction:: maia.algo.dist.convert_elements_to_mixed
.. autofunction:: maia.algo.dist.convert_mixed_to_elements
.. autofunction:: maia.algo.dist.reorder_elt_sections_from_dim
.. autofunction:: maia.algo.dist.concatenate_elt_sections
.. autofunction:: maia.algo.dist.generate_jns_vertex_list
.. autofunction:: maia.algo.dist.find_ridges


Geometry transformations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.duplicate_from_periodic_jns
.. autofunction:: maia.algo.dist.duplicate_family_from_periodic_jns
.. autofunction:: maia.algo.dist.extrude
.. autofunction:: maia.algo.dist.agglomerate_cells
.. autofunction:: maia.algo.dist.merge_zones
.. autofunction:: maia.algo.dist.merge_zones_from_family
.. autofunction:: maia.algo.dist.merge_connected_zones
.. autofunction:: maia.algo.dist.remove_degen_faces_from_family
.. autofunction:: maia.algo.dist.adapt_mesh_with_feflo

Interface tools
^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.connect_1to1_families
.. autofunction:: maia.algo.dist.find_joins_donor_name
.. autofunction:: maia.algo.dist.enforce_symmetric_joins
.. autofunction:: maia.algo.dist.conformize_jn_pair

Data management
^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.redistribute_tree
.. autofunction:: maia.algo.dist.concatenate_subsets_from_families
.. autofunction:: maia.algo.dist.deconcatenate_subsets_from_families

..
  from .extract_surf_dmesh     import extract_surf_tree_from_bc

.. _user_man_part_algo:

Partitioned algorithms
----------------------

The following algorithms apply on maia partitioned trees.

Geometric calculations
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.compute_wall_distance

Mesh extractions
^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.iso_surface
.. autofunction:: maia.algo.part.plane_slice
.. autofunction:: maia.algo.part.spherical_slice
.. autofunction:: maia.algo.part.extract_part_from_zsr
.. autofunction:: maia.algo.part.extract_part_from_bc_name
.. autofunction:: maia.algo.part.extract_part_from_family

Interpolations
^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.centers_to_nodes
.. autofunction:: maia.algo.part.nodes_to_centers

.. _user_man_gen_algo:


Generic algorithms
------------------

The following algorithms apply on maia distributed or partitioned trees

Geometry transformations
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: maia.algo.transform_affine
.. autofunction:: maia.algo.scale_mesh
.. autofunction:: maia.algo.cartesian_to_cylindrical
.. autofunction:: maia.algo.cylindrical_to_cartesian

Geometric calculations
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: maia.algo.localize_points
.. autofunction:: maia.algo.find_closest_points
.. autofunction:: maia.algo.compute_elements_center
.. autofunction:: maia.algo.compute_elements_measure
.. autofunction:: maia.algo.compute_elements_normal

Interpolations
^^^^^^^^^^^^^^
.. autofunction:: maia.algo.interpolate

Connectivities conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: maia.algo.pe_to_nface
.. autofunction:: maia.algo.nface_to_pe
.. autofunction:: maia.algo.edge_pe_to_ngon
.. autofunction:: maia.algo.ngon_to_edge_pe


Sequential algorithms
---------------------

The following algorithms apply on regular (full) pytrees. Note that
these compatibility functions are also wrapped in the ``maia_poly_old_to_new``
and ``maia_poly_new_to_old`` scripts, see :ref:`Quick start<quick_start_req>` section.

.. autofunction:: maia.algo.seq.poly_new_to_old
.. autofunction:: maia.algo.seq.poly_old_to_new
.. autofunction:: maia.algo.seq.enforce_ngon_pe_local
