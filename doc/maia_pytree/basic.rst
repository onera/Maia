.. _pt_node_edit:

Basic node editing
==================

This page describe the ``maia.pytree`` functions to read or write
the different attributes of a CGNSTree. These functions are simple
but provide the basis for complex operations.

Vocabulary
----------

.. link sids to python

According to the CGNS standard, a CGNS **node** is a structure described by three attributes:

- a **name**, which is a case-sensitive identifier shorter than 32 characters;
- a **label**, which must be taken from a predefined list;
- a optional **value**, which is a (possibly multidimensional) array of data;

plus a list of **children** which are themselves nodes. Due to that,
nodes have in fact a hierarchic structure, which is why we rather employ the word
**tree** to refer to them.

The organisation of this structure (for example: what are the allowed labels
under each node, or what are the allowed values for each label)
is defined by the `SIDS
<https://cgns.github.io/CGNS_docs_current/sids/index.html>`_
and will not be described here.

In addition, the `sids-to-python specification
<https://cgns.github.io/CGNS_docs_current/python/index.html>`_
defines how to describe a node structure in python.
``maia.pytree`` conforms to this mapping which, in short, states that:

- node names and labels are simple ``str`` objects
- node values can be either ``None``, or a numpy f-contiguous array
- nodes are defined by a list following the pattern ``[name, value, children_list, label]``

Node edition
------------

Accessing to the different attributes of a node with the ``[]`` operator
is possible but error-prone. Consequently, we advice the user to use
the available get and set functions:

.. autosummary::
  ~maia.pytree.get_name
  ~maia.pytree.set_name
  ~maia.pytree.get_label
  ~maia.pytree.set_label
  ~maia.pytree.get_children
  ~maia.pytree.add_child
  ~maia.pytree.set_children
  ~maia.pytree.get_value
  ~maia.pytree.set_value
  ~maia.pytree.update_node

Here is few examples using theses functions:

.. code-block:: python

  >>> node = ["MyNode", None, [], "UserDefinedData_t"]
  >>> PT.get_name(node)
  'MyNode'
  >>> PT.set_value(node, [1,2,3])
  >>> PT.get_value(node)
  array([1, 2, 3], dtype=int32)
  >>> pnode = ["ParentNode", np.array([3.14]), [], "UserDefinedData_t"]
  >>> PT.set_children(pnode, [node])
  >>> len(PT.get_children(pnode))
  1
  >>> PT.update_node(node, name="MyNodeUpdated")
  >>> [PT.get_name(n) for n in PT.get_children(pnode)]
  ['MyNodeUpdated']

Similarly, although trees can be displayed with the :func:`print` function,
it is preferable to use :func:`~maia.pytree.print_tree` for a better rendering:

  >>> print(pnode)
  >>> print(pnode)
  ['ParentNode', array([3.14]), [['MyNodeUpdated', array([1, 2, 3], dtype=int32), [], 'UserDefinedData_t']], 'UserDefinedData_t']
  >>> PT.print_tree(pnode)
  ParentNode UserDefinedData_t R8 [3.14]
  └───MyNodeUpdated UserDefinedData_t I4 [1 2 3]

.. seealso:: In practice, it is common to use :ref:`searches <pt_node_search>`
  to navigate in tree hierarchie, and :ref:`inspection <pt_inspect>` to get
  relevant data.

Node creation
-------------

In the same idea of avoiding to manipulate the underlying list by hand, the following functions
can be used to create new nodes: 

.. autosummary::
  ~maia.pytree.new_node
  ~maia.pytree.new_child
  ~maia.pytree.update_child

The previous snippet can thus be rewritted in more compact form:

  >>> pnode = PT.new_node("ParentNode", "UserDefinedData_t", 3.14)
  >>> node = PT.new_child(pnode, "MyNode", "UserDefinedData_t", [1,2,3])
  >>> PT.update_node(node, name="MyNodeUpdated")
  >>> PT.print_tree(pnode)
  ParentNode UserDefinedData_t R8 [3.14]
  └───MyNodeUpdated UserDefinedData_t I4 [1 2 3]

.. seealso:: In practice, it is common to use :ref:`presets <pt_presets>` for a quicker
  and SIDS-compliant creation of nodes with a specific label.

API reference
-------------

.. autofunction:: maia.pytree.get_name
.. autofunction:: maia.pytree.get_label
.. autofunction:: maia.pytree.get_value
.. autofunction:: maia.pytree.get_value_type
.. autofunction:: maia.pytree.get_value_kind
.. autofunction:: maia.pytree.get_children

.. autofunction:: maia.pytree.set_name
.. autofunction:: maia.pytree.set_label
.. autofunction:: maia.pytree.set_value
.. autofunction:: maia.pytree.set_children
.. autofunction:: maia.pytree.add_child

.. autofunction:: maia.pytree.update_node

.. autofunction:: maia.pytree.new_node
.. autofunction:: maia.pytree.new_child
.. autofunction:: maia.pytree.update_child

.. autofunction:: maia.pytree.print_tree