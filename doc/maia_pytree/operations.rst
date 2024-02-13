.. _pt_operations:

Tree operations
===============

This page describe some ``maia.pytree`` features acting on whole CGNSTrees, such as
copying, displaying of comparaing trees.
These features are less SIDS-aware than the one listed
in :ref:`pt_inspect` or :ref:`pt_presets` pages.

Tree construction
-----------------

.. autofunction:: maia.pytree.shallow_copy
.. autofunction:: maia.pytree.deep_copy

Tree comparisons
----------------

``maia.pytree`` offers two ways to compare trees. We can either simply check
if two trees are identical or not with the functions 
:func:`~maia.pytree.is_same_node` and :func:`~maia.pytree.is_same_tree`, 
or get a full report of differences using :func:`~maia.pytree.diff_tree`.
The later also allow users to define their own comparison function
for data arrays. 

Checking trees equality
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.pytree.is_same_node
.. autofunction:: maia.pytree.is_same_tree

Reporting trees differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maia.pytree.diff_tree

.. rubric:: Comparison methods

The following comparison methods are exposed in the ``maia.pytree.compare`` module:

.. autofunction:: maia.pytree.compare.EqualArray
.. autofunction:: maia.pytree.compare.CloseArray

To define a custom comparison method, users must provide a *callable* object comparing two
nodes. Note that this method will only be called on nodes having numerics (excluding str)
values of same shape, since :func:`diff_tree` will report a difference otherwise.
This callback must have the following signature: 

.. code-block::

  def __call__(nodes_1: List[CGNSTree], 
               nodes_2: List[CGNSTree]) -> Tuple[bool,str,str]: 

where ``nodes_1`` and ``nodes_2`` are the two nodes on which comparison is performed,
stacked with their ancestors.

As an example, here is a dummy comparison method definition:

.. code-block::

  class CustomComparator:
    def __call__(self, nodes_1, nodes_2):
      if 'BC_t' in [PT.get_label(n) for n in nodes_1]:
        return (True, '', '') # If node is below a BC, considerer it's ok
      else:
        return (False, 'Uncomparable value\n', '')

.. rubric:: Comparison reports

Comparison reports are named tuple made of 3 values: 

.. autoclass:: maia.pytree.compare.DiffReport

The choice to write in errors or warning report is let to the comparison object.
To illustrate the content of a ``DiffReport``, consider the following example:

  >>> zone1 = PT.yaml.to_node('''
    Zone Zone_t I4 [[9,4,0]]:
      ZoneType ZoneType_t "Unstructured":
      FlowSolution FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        Density DataArray_t [1., 1, 1, 1]:
        Pressure DataArray_t [1E5, 1E5, 1E5, 1E5]:
      SomeData ZoneSubRegion_t:
        PointList IndexArray_t [2,4,6,8]:
        Descriptor Descriptor_t "Even vtx":
    ''')
  >>> zone2 = PT.yaml.to_node('''
    Zone Zone_t I8 [[9,4,0]]:
      ZoneType ZoneType_t "Unstructured":
      FamilyName FamilyName_t "ROTOR":
      FlowSolution FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        Density DataArray_t [1., 1.1, 1, 1]:
      SomeData DiscreteData_t:
        PointList IndexArray_t [1,3,5,7,9]:
        Descriptor Descriptor_t "Odd vtx":
    ''')
  >>> report = PT.diff_tree(zone1, zone2)
    
Printing the errors report with ``>>> print(report.errors)`` gives

.. code-block:: python

  /Zone -- Value types differ: I4 <> I8
  /Zone/FlowSolution/Density -- Values differ: [1. 1. 1. 1.] <> [1.  1.1 1.  1. ]
  < /Zone/FlowSolution/Pressure
  /Zone/SomeData -- Labels differ: ZoneSubRegion_t <> DiscreteData_t
  /Zone/SomeData/Descriptor -- Values differ: Even vtx <> Odd vtx
  /Zone/SomeData/PointList -- Value shape differ: (4,) <> (5,)
  > /Zone/FamilyName

Following ``diff`` convention, symbols ``<`` (resp. ``>``) are used to mark nodes
existing only in first (resp. second) tree.
Nodes existing in both trees, but with different label or values are shown using
the following pattern : ``{path} -- {diff_kind}: {details}``.
If the ``<>`` symbol is used to display details,
the value to its left (resp. right) corresponds to the first (resp. second) input node.

.. Note:: Users implementing their own ``comp`` object must only write the equivalent
  of ``{details}`` in the report. Other data will be insered automatically.