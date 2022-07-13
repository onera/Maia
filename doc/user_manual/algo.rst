Algo module
===========

The ``maia.algo`` module provides various parallel operations to be applied to one of the
two kind of trees defined by maia:

- ``maia.algo.dist`` module contains some operations applying on distributed trees
- ``maia.algo.part`` module contains some operations applying on partitioned trees

In addition, few algorithms can be applied indistinctly to distributed or partitioned trees.
These algorithms are accessible through the ``maia.algo`` module.

.. _user_man_dist_algo:

Dist algo
---------

The following algorithms applies on maia distributed trees.


Connectivities conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.convert_s_to_u
.. autofunction:: maia.algo.dist.generate_ngon_from_std_elements
.. autofunction:: maia.algo.dist.convert_ngon_to_std_elements
.. autofunction:: maia.algo.dist.generate_jns_vertex_list


Geometry transformations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.algo.dist.duplicate_zone_with_transformation
.. autofunction:: maia.algo.dist.duplicate_from_rotation_jns_to_360
.. autofunction:: maia.algo.dist.merge_zones
.. autofunction:: maia.algo.dist.merge_connected_zones
.. autofunction:: maia.algo.dist.conformize_jn_pair

Mesh extractions
^^^^^^^^^^^^^^^^
..
  from .extract_surf_dmesh     import extract_surf_tree_from_bc

.. _user_man_part_algo:

Part algo
---------

The following algorithms applies on maia partitioned trees.

.. autofunction:: maia.algo.part.compute_cell_center
.. autofunction:: maia.algo.part.compute_wall_distance
.. autofunction:: maia.algo.part.localize_points
.. autofunction:: maia.algo.part.find_closest_points
.. autofunction:: maia.algo.part.interpolate_from_part_trees

.. _user_man_gen_algo:

Generic algo
------------

The following algorithms applies on maia distributed or partitioned trees
