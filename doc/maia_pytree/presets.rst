.. _pt_presets:

Node creation presets
=====================

Although any tree could be creating using exclusivelly :func:`~maia.pytree.new_node` function,
``maia.pytree`` provide some shortcuts to create nodes with relevant information.

In addition of reducing the amount of code to write, it also:

- hides the node structure details : if you want to create an *Unstructured* zone,
  just tell to new_Zone function: you don't need to know that this information
  should be stored in a ZoneType node of label ZoneType_t node under the zone;
- performs checks to prevent you to create non CGNS/SIDS-compliant nodes.


Generalities
------------

.. _pt_presets_commun:

.. rubric:: Containers fields

When creating a *Container* node (ie. a node storing some fields,
such as a FlowSolution_t), a list of DataArray to create can be 
provided through the ``fields`` parameter, which must be a dictionnary
mapping array names to array values: for example,

>>> fields = {'Density' : np.array([1, 1, 1.05], float),
...           'Temperature' : [293., 293., 296.]}

when passed to :func:`~maia.pytree.node.presets.new_FlowSolution`,
will created the requested fields:

>>> fs = PT.new_FlowSolution('FS', fields=fields)
>>> PT.print_tree(fs)
FS FlowSolution_t 
├───Density DataArray_t R8 [1.   1.   1.05]
└───Temperature DataArray_t R4 [293. 293. 296.]

Notice that values which are not numpy array instance are converted following
:func:`~maia.pytree.set_value` rules.

.. rubric:: Parent label check

All the :func:`new_...` functions (appart from :func:`new_CGNSTree`) take an optionnal
``parent`` argument, which can be used to add the created node
to the parent children list. In this case, a warning will be issued if the hierarchic
relation is not CGNS/SIDS-compliant.

>>> zone = PT.new_Zone()
>>> bc = PT.new_BC(parent=zone)
RuntimeWarning: Attaching node BC (BC_t) under a Zone_t parent
is not SIDS compliant. Admissible parent labels are ['ZoneBC_t'].


.. rubric:: Return value

.. important:: All the functions listed in this page return a single value,
  which is the created CGNSTree. For more readability, we omit the return section
  in the API description.
  
Overview
--------

*Functions creating top level structures*

.. autosummary::
  :nosignatures:

  ~maia.pytree.node.presets.new_CGNSTree
  ~maia.pytree.node.presets.new_CGNSBase
  ~maia.pytree.node.presets.new_Zone

*Functions creating Family related nodes*

.. autosummary::
  :nosignatures:

  ~maia.pytree.node.presets.new_Family
  ~maia.pytree.node.presets.new_FamilyName

*Functions creating Zone related nodes*

.. autosummary::
  :nosignatures:

  ~maia.pytree.node.presets.new_GridCoordinates

  ~maia.pytree.node.presets.new_Elements
  ~maia.pytree.node.presets.new_NFaceElements
  ~maia.pytree.node.presets.new_NGonElements

  ~maia.pytree.node.presets.new_ZoneBC
  ~maia.pytree.node.presets.new_ZoneGridConnectivity

  ~maia.pytree.node.presets.new_FlowSolution
  ~maia.pytree.node.presets.new_ZoneSubRegion

  ~maia.pytree.node.presets.new_BC
  ~maia.pytree.node.presets.new_GridConnectivity
  ~maia.pytree.node.presets.new_GridConnectivity1to1
  ~maia.pytree.node.presets.new_GridConnectivityProperty

*Functions creating common sub nodes*

.. autosummary::
  :nosignatures:

  ~maia.pytree.node.presets.new_GridLocation
  ~maia.pytree.node.presets.new_DataArray
  ~maia.pytree.node.presets.new_IndexArray
  ~maia.pytree.node.presets.new_IndexRange
  

API reference
-------------

.. automodule:: maia.pytree.node.presets
  :members:
