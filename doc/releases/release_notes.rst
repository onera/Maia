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
- Algo module: add ``extract_part_from_zsr`` API to extract a submesh from FamilyName nodes
- Algo module: add ``scale_mesh`` to scale the coordinates of a cartesian mesh
- Algo module: add ``nodes_to_centers`` to interpolate FlowSolution_t from Vertex to CellCenter

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- connect_1to1_families: manage Elements meshes
- extract_part_from_xxx: transfer BCs on extracted meshes and manage structured meshes
- partitioning: enable split of 2D/1D structured meshes
- interpolation: allow input fields to be Vertex located in some cases
- poly_old_to_new / poly_new_to_old : manage *mixed* elements
- convert_elements_to_ngon : manage *mixed* elements

🐞 Fixes
^^^^^^^^
- merge_zones: manage S/U GridConnectivity_t when merging U zones
- add_joins_donor_name: prevent a crash when some GCs already have their DonorName
- transform_affine : manage partitioned S zones and 2D meshes
- transfer module : prevent a bug occurring when subset nodes have a dot in their name
- convert_mixed_to_elements: prevent a bug occurring when multiple MIXED nodes are used

🚧 API change
^^^^^^^^^^^^^
- extract_part_from_zsr: add ``transfert_dataset`` argument for easier transfer of current ZSR fields
- convert_s_to_u: operate inplace (input tree is modified). Will return None in next release.

v1.2 (July 2023)
----------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``adapt_mesh_with_feflo``, wrapping *Feflo.a* to perform mesh adaptation
- Factory module : add ``dist_to_full_tree`` to gather a distributed tree into a standard tree
- File management: add ``read_links`` function to get the links from a CGNS file
- File management: add ``file_to_part_tree`` function to read maia partitioned trees

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
- distributed io: fix read/write of S meshes for data smaller than comm size
- elements to ngon conversion: manage vertex located BCs

🚧 API change
^^^^^^^^^^^^^
- redistribute_tree: remove default value for policy
- wall_distance: remove families parameter
- ``distribute_tree`` renamed into ``full_to_dist_tree``

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
