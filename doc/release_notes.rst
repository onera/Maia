.. _release_notes:

Release notes
=============

.. _whatsnew:

.. currentmodule:: maia

This page contains information about what has changed in each new version of **Maia**.

Development version
-------------------

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- VStrideArray: implement OUTER_AXIS mode for ``sort`` and ``unique`` functions
- maia_cgns_check: introduce stage 3 including some topological checks

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``agglomerate_cells`` to create structured coarsened meshes
- Algo module: add ``enforce_symmetric_joins`` to have a simplier description of matching interfaces

🐞 Fixes
^^^^^^^^
- centers_to_nodes: add checks and fix empty partition bugs
- PT.get_node_from_predicates: fixup behaviour (admissible node was sometime not found)
- pe_to_nface: fix a crash occuring when a rank locally owns only one face
- extract_part_from_family: fixup concatenation order of fields when ``data_transfer=True``
- adapt_mesh_with_feflo: fix failure on 2D meshes by adding default 2D elements tag
- merge_zones: prevent a dtype issue on 2D meshes

v1.8 (September 2025)
---------------------

💡 New Features
^^^^^^^^^^^^^^^
- Introduce command line utility ``maia_cgns_check`` to detect issues in cgns files
- Pytree: add ``visit`` and ``scan`` functions to apply a callable to every node of the tree

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- extract_part: add local mode for unstructured meshes
- maia_print_tree: trees are loaded much faster, which allows to print large files
- generate_dist_block: manage non uniform Nx,Ny,Nz for Poly case
- connect_1to1_families: extend to 2D input meshes
- generate_jns_vertex_list, merge_zones: extend to polyedric 2D input meshes
- extract_part: extend to 2D input meshes
- transform affine: detect all vectorial fields if ``apply_to_fields`` is used
- Introduce ``'ALL'`` shortcut as admissible value for ``containers_name`` in relevant functions

🐞 Fixes
^^^^^^^^
- extract_part: fix output cell GlobalNumbering of surfacic S extractions and allow any input basename
- maia_poly_old_to_new/maia_poly_new_to_old: prevent overflows when converting large meshes
- convert_mixed_to_elements: correct BC conversion on meshes having multiple MIXED nodes
- convert_s_to_u: fix jmax boundary faces for standard elements output
- connect_1to1_families: prevent internal faces to be wrongly report as unmatched
- PT.diff_tree: prevent deadlock when input trees have a different distribution
- concatenate_subsets_from_families: prevent a crash when families do not span over all zones
- iso_surface & slices: fix crash occuring during non-R8 partial fields exchanges in specific cases

🔧 Advanced users / devs
^^^^^^^^^^^^^^^^^^^^^^^^
- Introduce the ``maia.pytree.pred`` module to facilitate creation of node seaching predicates

🚧 API change
^^^^^^^^^^^^^
- pytree: Element.Type now return the ElementType value (*eg* TRI_3) instead of enumeration number (*eg* 5)
- pytree: Subset.ZSRExtent and the whole Subset.BCDataSet namespace are deprecated
- adapt_mesh_with_feflo: change ``container_names`` argument into ``containers_name``

v1.7 (May 2024)
---------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``compute_elements_normal`` to compute face or edge normal (depending on PhyDim)
- Support for type hints

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- Manage preexisting std / mixed Elements_t nodes in std ↔ mixed conversion
- Update Periodic_t values in cartesian ↔ cylindrical conversions
- localize_points: allow source mesh to be structured
- Allow distributed meshes in localize_points and find_closest_points
- Allow distributed meshes in interpolation
- convert_s_to_u: manage 2D input meshes and add standard elements output mode
- IO functions: automatically create non existing folders in writting functions
- Create internal edges in isosurface and extract_part functionnalities
- Create internal 1to1-GridConnectivity nodes in extract_part functionnality

🐞 Fixes
^^^^^^^^
- Pytree / YAML loader: allow spaces in nodes name
- concatenate_subsets_from_families: fix GridLocation value of updated ZoneSubRegion nodes
- compute_wall_distance: write output (using huge value) even if no BCWall are found in mesh
- extract_part: return an empty tree instead of raising when requested subset does not exist
- convert_s_to_u: correct PointList values of CellCenter subsets
- merge_zones: automatically create NGON/ParentElements if not existing
- merge_zones: prevent too long node names for output JNs when ``concatenate_jns=True``
- Better management of phyDim=2 in some functions

🔧 Advanced users / devs
^^^^^^^^^^^^^^^^^^^^^^^^
- Introduce the ``VStrideArray`` class to facilitate connectivity array manipulation
- Add options to manage multiple occurences in the ``Put`` method of ``Global(Multi)Indexer``

v1.6 (January 2025)
-------------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``(de)concatenate_subsets_from_family`` functions to (un)gather BC nodes
- Algo module: add ``extrude`` function to create 3D meshes from 2D surfacic meshes

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- generate_dist_block: add element kind ``BAR_2`` to generate lineic 1D meshes
- generate_dist_block: generate blocks aligned with custom basis vectors
- partition_dist_tree: manage ZoneSubRegion nodes when splitting structured meshes
- extract_part: also expose extractor object in ``bc_name`` and ``family`` APIs
- extract_part: preserve BC kind and metadata in 3D extraction
- convert_ngon_to_elements: manage 2D cases
- Better management of hybrid meshes in NGon ↔ Element conversions

🐞 Fixes
^^^^^^^^
- merge_zones: copy AdditionalFamilyName nodes when merging subsets
- recover_dist_tree: fix a bug in NFace values and manage missing PE case
- file_to_part_tree: fix an error occuring when ``redispatch=True`` and remove too agressive checks
- wall_distance: fix case ``periodic=True`` on meshes having unconnected zones groups
- interpolate: prevent a crash when a rank does not hold any partition on source tree
- redistribute_tree and full_to_dist_tree : fix an exception occuring on 2D structured meshes
- connect_1to1_families: prevent a ``ZeroDivisionError`` when no connections are found
- duplicate_from_rotation_jns_to_360: add a tolerance to manage close to zero rotation angles
- part_tree_to_dist_tree_copy: manage data under GridConnectivity_t nodes
- convert_ngon_to_elements: preserve internal edges (in 2D) or faces (in 3D) indexed by a subset

🚧 API change
^^^^^^^^^^^^^
- generate_dist_block: rename parameter ``edge_length`` into ``length``
- Flag ``legacy=True`` for IO functions is now ignored, and will be removed in next release

v1.5 (September 2024)
---------------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``edge_pe_to_ngon``/``ngon_to_edge_pe`` to convert 2D polyhedric connectivities
- Algo module: add ``find_ridges`` functionnality to retrieve topological ridges between surfaces
- Algo module: add ``compute_elements_measure`` to compute length, area or volume of mesh entities
- Algo module: add ``remove_degen_faces_from_family`` to cleanup faces with zero surface

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- dist_tree_to_file, part_tree_to_file, write_trees: write user provided links
- part_tree_to_file: write CGNSBase_t misc. children
- interpolate: managed Vertex located fields for all values of ``strategy`` parameter
- adapt_mesh_with_feflo: allow users to change temporary directory for meshb files
- recover_dist_tree: manage poly 2D (NGON+BAR) meshes
- Introduce ``compute_elements_center`` which allows more configurations for centers computing
- merge_zones: add ``family`` option for subset merge parameter
- rearrange_element_sections: split reordering and concatenation in two functions (flexibility)
- localize_points, compute_wall_distance: allow 2D polygonal or Elements meshes
- ngons_to_elements: functionnality is now truly parallel, and more nodes are managed
- partitioning, recover_dist_tree: add ``data_transfer`` shortcut to transfer fields or UDD nodes

🐞 Fixes
^^^^^^^^
- transform_affine: update periodic values of GridConnectivity_t nodes
- extract_part: prevent a crash when a rank get no cells on extracted U zone
- extract_part: fix missing fields problem occuring on some 2D structured extractions
- recover_dist_tree: prevent a crash when a partitioned CGNSBase_t does not exist on all ranks
- partitioning : fix internal edges creation and BCs on 2D/NGon meshes
- partitioning : fix PointList values of hybrid U/S GridConnectivity_t nodes
- partitioning: manage ``DiscreteData_t`` nodes with PointList
- convert_elements_to_ngon: preserve CellCenter BC_t nodes during conversion
- file_to_dist_tree: fix read of structured BCs having both PointList and BCDataSet children
- **[v1.5.1]** ensure compatibility with ParaDiGM 2.6.0

🚧 API change
^^^^^^^^^^^^^
- dist_tree_to_file, part_tree_to_file & write_trees: add ``links`` parameter
- Rename ``ngons_to_elements`` into ``convert_ngon_to_elements``
- Remove direct calls ``compute_{cell|face|edge}_center`` in favor of ``compute_elements_center``
- Split ``rearrange_element_sections`` into ``reorder_elt_sections_...`` + ``concatenate_elt_sections``


v1.4 (May 2024)
---------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``cartesian_to_cylindrical``/``cylindrical_to_cartesian`` to change axis system
- Transfer module: add two functions to copy nodes between distributed and partitioned trees

🚀 Feature improvements
^^^^^^^^^^^^^^^^^^^^^^^
- adapt_mesh_with_feflo: manage axisymmetric meshes when ``perio=True``
- adapt_mesh_with_feflo: manage general U/elt meshes in meshb converter
- compute_wall_distance: use volumic face global numbering in ClosestEltGnum result
- closest_points: allow meshes of any cell dimension
- Manage ArbitraryGridMotion_t nodes in IO, split and data transfer
- Manage global BCData_t arrays in IO, split and data transfer
- Add shortcut functions to duplicate zones belonging to a family

🐞 Fixes
^^^^^^^^
- extract_part: ensure dtype correctness of transfered arrays
- extract_part_from_bc_name: prevent crash if BCDataSet has a PointList or PointRange
- extract_part_from_family: prevent a crash if a BC name is identical to the requested family
- recover_dist_tree: prevent a crash if some Elements_t nodes does not exist in every zones
- recover_dist_tree: use gnum_dtype as output kind of created zones
- convert_elements_to_ngon: update FaceCenter / CellCenter PointList arrays
- partitioning: fix creation of cells global numbering for multisection element meshes
- merge_zones: fix type mismatch occurring on I4 meshes with I8 production
- **[v1.4.1]** mesh adaptation: fix tensorial metric and trees with multiple Elements_t of same type
- **[v1.4.1]** merge_zones: prevent a crash when zones have non 1to1 GridConnectivity nodes
- **[v1.4.2]** file_to_dist_tree: restore Python 3.7 compatibility

🚧 API change
^^^^^^^^^^^^^
- default value for ``apply_to_fields`` is now ``True`` in exposed functions
- convert_s_to_u: now return ``None`` as announced
- Change exposed functions in ``maia.pytree`` (refer to related documentation)

v1.3 (January 2024)
-------------------

💡 New Features
^^^^^^^^^^^^^^^
- Algo module: add ``extract_part_from_family`` API to extract a submesh from FamilyName nodes
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
- adapt_mesh_with_feflo: add an option to manage periodic meshes

🐞 Fixes
^^^^^^^^
- merge_zones: manage S/U GridConnectivity_t when merging U zones
- add_joins_donor_name: prevent a crash when some GCs already have their DonorName
- transform_affine : manage partitioned S zones and 2D meshes
- transfer module : prevent a bug occurring when subset nodes have a dot in their name
- convert_mixed_to_elements: prevent a bug occurring when multiple MIXED nodes are used
- **[v1.3.1]** io: fix read of unstructured BCDataSet having a PointList

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
