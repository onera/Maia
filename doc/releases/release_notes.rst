.. _release_notes:

Release notes
=============

.. _whatsnew:

.. currentmodule:: maia

This page contains information about what has changed in each new version of **Maia**.

Developpement version
---------------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add a function wrapping *Feflo.a* to perform mesh adaptation

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- file_to_dist_tree: correct unsigned NFace connectivity if possible
- wall_distance: add an option to take into account periodic connectivities 
- poly_old_to_new / poly_new_to_old : support 2D meshes

🐞 Fixes
^^^^^^^^
- merge_zones: fix unwanted merge of BCDataSet_t when merge_strategy is None
- partitioning: fix global numbering of S BCDataSet + fix GC-related ZGC 
- isosurface: fix poor performances + better management of corner cases

🚧 API change
^^^^^^^^^^^^^
- redistribute_tree: remove default value for policy
- wall_distance: remove families parameter

🔧 Advanced users / devs
^^^^^^^^^^^^^^^^^^^^^^^^
- Add a method to give a global id to any object in parallel

v1.1 (May 2023)
---------------

💡 New Features
^^^^^^^^^^^^^^^

- Algo module: generate (periodic) 1to1 GridConnectivity between selected BC or GC
- Factory module: generate 2D spherical meshes and points clouds

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- generate_dist_block: enable generation of structured meshes
- partitioning: enable split of 2D (NGON/Elts) and 1D (Elts) meshes
- partitioning: copy AdditionalFamilyName and ReferenceState from BCs to the partitions
- compute_face_center : manage structured meshes
- merge_zones: allow wildcards in zone_paths
- isosurface: recover volumic GCs on surfacic tree (as BCs)
- transfer (part->dist): manage BC/BCDataSet created on partitions for structured meshes

🐞 Fixes
^^^^^^^^
- convert_elements_to_ngon: prevent a memory error & better management of 2D meshes
- isosurface: improve robustness of edge reconstruction
- partitioning: fix split of structured GCs and BCDataSet
- merge_zone: fix a bug occurring when FamilyName appears under some BC_t nodes

🔧 Advanced users / devs
^^^^^^^^^^^^^^^^^^^^^^^^
- use new pytest_parallel module
- transfer (part->dist): add user callback to reduce shared entities


v1.0 (March 2023)
-----------------
First release of Maia !
