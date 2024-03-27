Algo module
===========

The ``maia.algo`` module provides various algorithms to be applied to one of the
two kind of trees defined by Maia:

- ``maia.algo.dist`` module contains some operations applying on distributed trees
- ``maia.algo.part`` module contains some operations applying on partitioned trees

In addition, some algorithms can be applied indistinctly to distributed or partitioned trees.
These algorithms are accessible through the ``maia.algo`` module.

The ``maia.algo.seq`` module contains a few sequential utility algorithms.

.. _user_man_dist_algo:

Distributed algorithms
----------------------

The following algorithms applies on maia distributed trees.


Connectivities conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.convert_s_to_u
.. autofunction:: maia.algo.dist.convert_elements_to_ngon
.. autofunction:: maia.algo.dist.ngons_to_elements
.. autofunction:: maia.algo.dist.convert_elements_to_mixed
.. autofunction:: maia.algo.dist.convert_mixed_to_elements
.. autofunction:: maia.algo.dist.rearrange_element_sections
.. autofunction:: maia.algo.dist.generate_jns_vertex_list


Geometry transformations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.duplicate_from_rotation_jns_to_360
.. autofunction:: maia.algo.dist.merge_zones
.. autofunction:: maia.algo.dist.merge_zones_from_family
.. autofunction:: maia.algo.dist.merge_connected_zones
.. autofunction:: maia.algo.dist.conformize_jn_pair
.. autofunction:: maia.algo.dist.adapt_mesh_with_feflo

Interface tools
^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.connect_1to1_families

Data management
^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.redistribute_tree

..
  from .extract_surf_dmesh     import extract_surf_tree_from_bc

.. _user_man_part_algo:

Partitioned algorithms
----------------------

The following algorithms applies on maia partitioned trees.

Geometric calculations
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.compute_cell_center
.. autofunction:: maia.algo.part.compute_face_center
.. autofunction:: maia.algo.part.compute_edge_center
.. autofunction:: maia.algo.part.compute_wall_distance
.. autofunction:: maia.algo.part.localize_points
.. autofunction:: maia.algo.part.find_closest_points

Mesh extractions
^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.iso_surface
.. autofunction:: maia.algo.part.plane_slice
.. autofunction:: maia.algo.part.spherical_slice
.. autofunction:: maia.algo.part.extract_part_from_zsr
.. autofunction:: maia.algo.part.extract_part_from_bc_name

Interpolations
^^^^^^^^^^^^^^

.. autofunction:: maia.algo.part.interpolate_from_part_trees
.. autofunction:: maia.algo.part.centers_to_nodes

.. _user_man_gen_algo:


Generic algorithms
------------------

The following algorithms applies on maia distributed or partitioned trees

.. autofunction:: maia.algo.transform_affine
.. autofunction:: maia.algo.pe_to_nface
.. autofunction:: maia.algo.nface_to_pe


Sequential algorithms
---------------------

The following algorithms applies on regular pytrees.

.. autofunction:: maia.algo.seq.poly_new_to_old
.. autofunction:: maia.algo.seq.poly_old_to_new
.. autofunction:: maia.algo.seq.enforce_ngon_pe_local
