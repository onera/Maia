.. _pt_inspect:

Node inspection
===============

To avoid redoing several times the most commun operations, ``maia.pytree`` provide
several functions extracting usefull data from a specific kind of CGNSNode.

All these functions are CGNS/SIDS aware, meaning that they are garanteed to succeed
only if the input tree is respecting the standard.

They are also read only; i.e. input tree will not be modified by any
of these calls.

Overview
--------

Here is a summary of the available functions, depending of the input node:

**Tree** *These functions apply to CGNSTree_t node*

.. autosummary::
  :nosignatures:

  ~maia.pytree.Tree.find_connected_zones
  ~maia.pytree.Tree.find_periodic_jns

**Zone** *These functions apply to Zone_t node*

.. autosummary::
  :nosignatures:

  ~maia.pytree.Zone.Type
  ~maia.pytree.Zone.IndexDimension
  ~maia.pytree.Zone.VertexSize
  ~maia.pytree.Zone.FaceSize
  ~maia.pytree.Zone.CellSize
  ~maia.pytree.Zone.VertexBoundarySize
  ~maia.pytree.Zone.NGonNode
  ~maia.pytree.Zone.NFaceNode
  ~maia.pytree.Zone.CellDimension
  ~maia.pytree.Zone.n_cell
  ~maia.pytree.Zone.n_face
  ~maia.pytree.Zone.n_vtx
  ~maia.pytree.Zone.n_vtx_bnd
  ~maia.pytree.Zone.has_ngon_elements
  ~maia.pytree.Zone.has_nface_elements
  ~maia.pytree.Zone.coordinates
  ~maia.pytree.Zone.get_ordered_elements
  ~maia.pytree.Zone.get_ordered_elements_per_dim
  ~maia.pytree.Zone.get_elt_range_per_dim
  ~maia.pytree.Zone.elt_ordering_by_dim
  
**Element** *These functions apply to Element_t node*

.. autosummary::
  :nosignatures:

  ~maia.pytree.Element.CGNSName
  ~maia.pytree.Element.Dimension
  ~maia.pytree.Element.NVtx
  ~maia.pytree.Element.Range
  ~maia.pytree.Element.Size
  ~maia.pytree.Element.Type

**GridConnectivity** *These functions apply to GridConnectivity_t and GridConnectivity1to1_t nodes*

.. autosummary::
  :nosignatures:

  ~maia.pytree.GridConnectivity.Type
  ~maia.pytree.GridConnectivity.is1to1
  ~maia.pytree.GridConnectivity.isperiodic
  ~maia.pytree.GridConnectivity.ZoneDonorPath
  ~maia.pytree.GridConnectivity.periodic_values

**Subset** *These functions apply to nodes having a PointList or a PointRange*

.. autosummary::
  :nosignatures:

  ~maia.pytree.Subset.getPatch
  ~maia.pytree.Subset.GridLocation
  ~maia.pytree.Subset.ZSRExtent
  ~maia.pytree.Subset.n_elem


.. note:: Functions are displayed below as static methods, gathered into classes.
  This is an implementation detail to put functions into namespaces : they should
  be used as usual, with their name prefixed by the label name:

  >>> PT.Zone.Type(zone_node) # Apply on a Zone_t node
  >>> PT.GridConnectivity.Type(gc_node) #Apply on a GC_t or GC1to1_t node

Methods detail
--------------

.. autoclass:: maia.pytree.Tree
  :members: 
.. autoclass:: maia.pytree.Zone
  :members: 
.. autoclass:: maia.pytree.Element
  :members:
.. autoclass:: maia.pytree.GridConnectivity
  :members:
.. autoclass:: maia.pytree.Subset
  :members:


