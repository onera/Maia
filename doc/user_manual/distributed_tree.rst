.. contents:: :local:

.. _dist_tree:

Loading a distributed tree
==========================

A *dist tree* is a CGNS tree where the tree structure is replicated across all processes, but array values of the nodes may be distributed. 

:
The generalized paths of the distributed nodes are listed below. For all other nodes within the tree, the node values are loaded by all processes.

TODO: make it possible to add node generalized path to the list of node that are distributed
TODO: warn if loading array > 10000

TODO move this definition
A *generalized path* is a path where each token of the path is either a node *name* or a node *label*. Example: "Zone_t/Hexa" is a generalized path and a node will match if it is a sub-node of any zone, and if its name is "Hexa". 
Note: of course there is no way to distinguish if we mean "of type Zone_t" or "of name Zone_t" so it will match both.


.. code::
  CGNSBase_t
  |___Zone_t
     |___GridCoordinates_t
     |   |___DataArray_t
     |
     |___Elements_t
     |   |___ElementConnectivity
     |   |___ElementStartOffset
     |   |___ParentElement
     |   |___ParentElementPosition   [not implemented]
     |
     |___FlowSolution_t
     |   |___DataArray_t
     |   |___PointList
     |
     |___DiscreteData_t
     |   |___DataArray_t
     |   |___PointList
     |
     |___ZoneBC_t
     |   |___BC_t
     |       |___PointList
     |       |___BCDataSet_t
     |           |___PointList
     |           |___BCData_t
     |               |___DataArray_t
     |
     |___ZoneGridConnectivity
     |   |___GridConnectivity
     |       |___PointList
     |       |___PointListDonor
     |
     |___ZoneSubRegion
         |___DataArray_t
         |___PointList



For these nodes, array values are distributed across processes. That is, for a dist_tree, on one process, the value of the node is actually a sub-interval array of the whole array. For each node, the sub-interval, plus the total size of the array are given by the triplet `[sub_first_index,sub_last_index,total_size]`.

The triplet is stored in a node `PartialDistribution` of type `Distribution_t`. Since many arrays are of the same size, they share this distribution, and there is not need to duplicate it for each shared array. `Distribution_t` nodes go here:

.. code::
  CGNSBase_t
  |___Zone_t
     |___**Distribution_t Distribution**
     |   |___IndexArray_t Vertex
     |   |___IndexArray_t Cell
     |   |___IndexArray_t VertexBoundary
     |
     |___GridCoordinates_t
     |   |___DataArray_t
     |   |___**Distribution_t** (optionel)
     |       |___IndexArray_t Vertex
     |
     |___Elements_t
     |   |___ElementConnectivity
     |   |___ElementStartOffset
     |   |___ParentElement
     |   |___ParentElementPosition   [not implemented]
     |   |___**Distribution_t**
     |       |___IndexArray_t Element
     |       |___IndexArray_t ElementStartOffset
     |
     |___FlowSolution_t
     |   |___DataArray_t
     |   |___PointList
     |   |___**Distribution_t** (si PointList et si pas de NFace)
     |   |___**Distribution_t**
     |       |___IndexArray_t Vertex|Cell|Face (Face si PointList)
     |
     |___DiscreteData_t
     |   |___DataArray_t
     |   |___PointList
     |   |___**Distribution_t** (si PointList)
     |       |___IndexArray_t Vertex|Cell|Face (Face si PointList)
     |
     |___ZoneBC_t
     |   |___BC_t
     |       |___PointList
     |       |___**Distribution_t**
     |           |___IndexArray_t Vertex|Cell|Face (Suivant GridLocation)
     |       |___BCDataSet_t
     |           |___PointList
     |               |___IndexArray_t Vertex|Cell|Face (Suivant GridLocation)
     |           |___**Distribution_t**
     |           |___BCData_t
     |               |___DataArray_t
     |
     |___ZoneGridConnectivity
     |   |___GridConnectivity
     |       |___PointList
     |       |___PointListDonor
     |       |___**Distribution_t**
     |           |___IndexArray_t Vertex|Cell|Face (Suivant GridLocation)
     |
     |___ZoneSubRegion               [not implemented]


.. code::
  CGNSBase_t
  |___Zone_t
     |___**GlobalNumbering**
     |   |___Vertex
     |   |___Cell
     |   |___CellBoundary
     |
     |___GridCoordinates_t
     |   |___DataArray_t
     |   |___**GlobalNumbering**
     |
     |___Elements_t
     |   |___ElementConnectivity
     |   |___ElementStartOffset
     |   |___ParentElement
     |   |___ParentElementPosition   [not implemented]
     |   |___**GlobalNumbering**
     |
     |___FlowSolution_t
     |   |___DataArray_t
     |   |___PointList
     |   |___**GlobalNumbering** (si PointList)
     |
     |___DiscreteData_t
     |   |___DataArray_t
     |   |___PointList
     |   |___**GlobalNumbering** (si PointList)
     |
     |___ZoneBC_t
     |   |___BC_t
     |       |___PointList
     |       |___**GlobalNumbering**
     |       |___BCDataSet_t
     |           |___PointList
     |           |___**GlobalNumbering**
     |           |___BCData_t
     |               |___DataArray_t
     |
     |___ZoneGridConnectivity
     |   |___GridConnectivity
     |       |___PointList
     |       |___PointListDonor
     |       |___**GlobalNumbering**
     |
     |___ZoneSubRegion               [not implemented]


Elements_t
----------

For heterogenous connectivities, the `ElementStartOffset` and `ElementConnectivity` arrays are not independent. The `ElementStartPartialDistribution` refers to the `ElementStartOffset` array (actually, the `ElementStartOffset` load one more element), and the partial ``ElementConnectivity` block loaded by one process is the one described by the `ElementStartOffset` block of that process.
  
TODO
ElementStartPartialDistribution
ElementConnectivityPartialDistribution

Example
-------

Let us look at this tree:

.. code:: yaml
  Base0 Base_t [3,3]:
    Zone0 Zone_t [[24],[6],[0]]:
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t [0,1,2,3,4,5,6]:
      Polygons Elements_t:
        ElementStartOffset DataArray_t [0,4,8]:
        ElementConnectivity DataArray_t [4,3,2,1, 1,5,6,7]:

TODO ajouter 2 BCs

If it is distributed on two processes, the dist_tree of each process will be:

.. code:: yaml
  Base0 Base_t [3,3]:
    Zone0 Zone_t [[24],[6],[0]]:
      GridCoordinates GridCoordinates_t:
        PartialDistribution Distribution_t [0,4,7]: # the block array contains data
                                                    # from sub-interval [0,3) and the array total size is 7 
        CoordinateX DataArray_t [0,1,2,3]:
      Polygons Elements_t:
        ElementStartPartialDistribution Distribution_t [0,2,3]: # the block array contains connectivities [0,1) (i.e. only 0)
        ElementStartOffset DataArray_t [0,4]: # in the global array, the connectivity starts at 0 and finishes at 4
        ElementConnectivity DataArray_t [4,3,2,1]: # this is connectivity 0

  Base0 Base_t [3,3]:
    Zone0 Zone_t [[24],[6],[0]]:
      GridCoordinates GridCoordinates_t:
        PartialDistribution Distribution_t [4,7,7]:
        CoordinateX DataArray_t [4,5,6]:
      Polygons Elements_t:
        ElementStartPartialDistribution Distribution_t [1,3,3]: # the block array contains connectivities [1,2) (i.e. only 1)
        ElementConnectivityPartialDistribution Distribution_t [4,8,8]: # the block array contains connectivities [1,2) (i.e. only 1)
        ElementStartOffset DataArray_t [4,8]: # in the global array, the connectivity starts at 4 and finishes at 8
        ElementConnectivity DataArray_t [1,5,6,7]: # this is connectivity 1


        Hexa Quad Tet

        LN_to_GN elements
        LN_to_GN cell
        LN_to_GN faces

        FlowSolution Hexa Tet
                     |0     10|10     20|
