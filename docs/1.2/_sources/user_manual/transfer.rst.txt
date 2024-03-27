Transfer module
===============

The ``maia.transfer`` contains functions that exchange data between the 
partitioned and distributed meshes. 

Fields transfer
---------------

High level APIs allow to exchange data at CGNS :cgns:`Tree` or :cgns:`Zone` level.
All the provided functions take similar parameters: a distributed tree (resp. zone),
the corresponding partitioned tree (resp. list of zones), the MPI communicator and
optionally a filtering parameter.

The following kind of data are supported: 
:cgns:`FlowSolution_t`, :cgns:`DiscreteData_t`, :cgns:`ZoneSubRegion_t` and :cgns:`BCDataSet_t`.

When transferring from distributed meshes to partitioned meshes, fields are supposed
to be known on the source mesh across all the ranks (according to :ref:`disttree definition<dist_tree>`).
Geometric patches (such as ZoneSubRegion or BCDataSet) must exists on the relevant partitions, meaning
that only fields are transfered.

When transferring from partitioned meshes to distributed meshes, geometric patches may or may not exist on the
distributed zone: they will be created if needed. This allows to transfer fields that have been created
on the partitioned meshes.
It is however assumed that global data (*e.g.* FlowSolution) are defined on every partition.

Tree level
^^^^^^^^^^

All the functions of this section operate inplace and require the following parameters:

- **dist_tree** (*CGNSTree*) -- Distributed CGNS Tree
- **part_tree** (*CGNSTree*) -- Corresponding partitioned CGNS Tree
- **comm**      (*MPIComm*)  -- MPI communicator

.. autofunction:: maia.transfer.dist_tree_to_part_tree_all
.. autofunction:: maia.transfer.part_tree_to_dist_tree_all

In addition, the next two methods expect the parameter **labels** (*list of str*) 
which allow to pick one or more kind of data to transfer from the supported labels.

.. autofunction:: maia.transfer.dist_tree_to_part_tree_only_labels
.. autofunction:: maia.transfer.part_tree_to_dist_tree_only_labels

Zone level
^^^^^^^^^^

All the functions of this section operate inplace and require the following parameters:

- **dist_zone**  (*CGNSTree*) -- Distributed CGNS Zone
- **part_zones** (*list of CGNSTree*) -- Corresponding partitioned CGNS Zones
- **comm**       (*MPIComm*)  -- MPI communicator

In addition, filtering is possible with the use of the
**include_dict** or **exclude_dict** dictionaries. These dictionaries map
each supported label to a list of cgns paths to include (or exclude). Paths
starts from the ``Zone_t`` node and ends at the targeted ``DataArray_t`` node.
Wildcard ``*`` are allowed in paths : for example, considering the following tree
structure,

.. code::

  Zone (Zone_t)
  ├── FirstSolution (FlowSolution_t)
  │   ├── Pressure (DataArray_t)
  │   ├── MomentumX (DataArray_t)
  │   └── MomentumY (DataArray_t)
  ├── SecondSolution (FlowSolution_t)
  │   ├── Pressure (DataArray_t)
  │   ├── MomentumX (DataArray_t)
  │   └── MomentumY (DataArray_t)
  └── SpecialSolution (FlowSolution_t)
      ├── Density (DataArray_t)
      └── MomentumZ (DataArray_t)

| ``"FirstSolution/Momentum*"`` maps to ``["FirstSolution/MomentumX", "FirstSolution/MomentumY"]``,  
| ``"*/Pressure`` maps to ``["FirstSolution/Pressure", "SecondSolution/Pressure"]``, and
| ``"S*/M*"`` maps to ``["SecondSolution/MomentumX", "SecondSolution/MomentumY", "SpecialSolution/MomentumZ"]``.

For convenience, we also provide the magic path ``['*']`` meaning "everything related to this
label".

Lastly, we use the following rules to manage missing label keys in dictionaries:

  - For _only functions, we do not transfer any field related to the missing labels;
  - For _all functions, we do transfer all the  fields related to the missing labels.

.. autofunction:: maia.transfer.dist_zone_to_part_zones_only
.. autofunction:: maia.transfer.part_zones_to_dist_zone_only
.. autofunction:: maia.transfer.dist_zone_to_part_zones_all
.. autofunction:: maia.transfer.part_zones_to_dist_zone_all
