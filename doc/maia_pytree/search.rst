.. _pt_node_search:

Node searching
==============

It is often necessary to select one or several nodes matching a specific
condition in a CGNSTree. ``maia.pytree`` provides various functions to
do that, depending of what is needed.

Tutorial
--------

Almost all the functions have the following pattern:

.. function:: get_node(s)_from_{predicate}(s)(node, condition, **kwargs)

``node`` has the same meaning for all the search functions, and is
simply the tree in which the search is performed.
Then, by choosing appropriate keyword for ``{predicate}`` and using
or not the ``s`` suffixes, lot of cases can be covered.

To illustrate the different possibilities, let's consider the following example
tree:

.. code-block::
  
  ZoneBC ZoneBC_t 
  ├───BC1 BC_t
  │   ├───GridLocation GridLocation_t "Vertex"
  │   ├───PointRange IndexRange_t [[ 1 11] [ 1  1]]
  │   └───BCDataSet BCDataSet_t "Null"
  │       ├───PointRange IndexRange_t [[1 5] [1 1]]
  │       └───GridLocation GridLocation_t "FaceCenter"
  └───BC2 BC_t 
      ├───GridLocation GridLocation_t "Vertex"
      └───PointRange IndexRange_t [[ 1 11] [ 6  6]]

Search condition
^^^^^^^^^^^^^^^^

The first customizable thing is ``{predicate}``, which indicates
which criteria is applied to every node to select it or not.
The ``condition`` is the concrete value of the criteria:

+--------------------------------+-----------------+
| Function                       | Condition kind  |
+================================+=================+
| :func:`get_..._from_name`      | str             |
+--------------------------------+-----------------+
| :func:`get_..._from_label`     | str             |
+--------------------------------+-----------------+
| :func:`get_..._from_value`     | value           |
+--------------------------------+-----------------+
| :func:`get_..._from_predicate` | function        |
+--------------------------------+-----------------+

The first functions are easy to understand: note just here that
``from_name`` and ``from_label`` accept wildcards ``*`` in their condition.

The last one is the more
general and take as predicate a function, which will be applied
to the nodes of the tree, and must return ``True`` (node is selected)
or ``False`` (node is not selected) : ``f(n:CGNSTree) -> bool`` 

Here is an example of these functions:

>>> PT.get_node_from_name(node, 'BC2')
# Return node BC2
>>> PT.get_node_from_label(node, 'BC_t')
# Return node BC1
>>> PT.get_node_from_value(node, [[1,5],[1,1]])
# Return the PointRange node located under BCDataSet
>>> PT.get_node_from_predicate(node, lambda n: 'BC' in PT.get_label(n)
...    and PT.Subset.GridLocation(n) == 'FaceCenter')
# Return node BCDataSet

.. seealso:: There is also a :func:`get_..._from_name_and_label` form, which takes two str
  as condition: first one is for the name, second one for the label.

  >>> PT.get_node_from_name_and_label(node, '*', '*Location_t')
  # Return node GridLocation of BC1

Number of results
^^^^^^^^^^^^^^^^^

The second customizable thing is the ``s`` after ``node``, which can be used to
decide if the search will return the first node matching the
predicate, or all the nodes matching the predicate:

+--------------------------------+--------------------------------------+
| Function                       | Return                               |
+================================+======================================+
| :func:`get_node_from_...`      | First node found or ``None``         |
+--------------------------------+--------------------------------------+
| :func:`get_nodes_from_...`     | List of all the nodes found or ``[]``|
+--------------------------------+--------------------------------------+

>>> PT.get_node_from_label(node, 'BC_t')
# Return node BC1
>>> PT.get_nodes_from_label(node, 'BC_t')
# Return a list containing BC1, BC2
>>> PT.get_node_from_label(node, 'DataArray_t')
# Return None
>>> PT.get_nodes_from_label(node, 'DataArray_t')
# Return an empty list


.. seealso:: All the :func:`get_nodes_from_...` functions have a
  :func:`iter_nodes_from_...` variant, which return a generator instead of a list
  and can be used for looping


Chaining searches (advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When looking for a particular node, it is often necessary to chain several
searches. For example, if we want to select the BCData nodes under a
BCDataSet node, we can do

>>> for bcds in PT.get_nodes_from_label(node, 'BCDataSet_t'):
>>>   for bcdata in PT.get_nodes_from_label(bcds, 'BCData_t'):
>>>     # Do something with bcdata nodes

This code can be reduced to a single function call by just adding a ``s`` to the search
function, and changing the ``condition`` to a list :

>>> for bcdata in PT.get_nodes_from_labels(node, ['BCDataSet_t', 'BCData_t']):
>>>   # Do something with bcdata nodes

Just as before, a search will be performed starting from ``node``, using the first
condition; from the results, a second search will be performed using the second
condition; and so on.

.. tip:: Since it can be cumbersome to write manually several predicate functions,
  the :func:`get_..._from_predicates` come with "autopredicate" feature: 
  the list of functions can be replaced by a string, which is converted from
  the following steps:

  - String is split from separator ``'/'``
  - Each substring is replaced by ``get_label`` if it ends with _t, 
    and by ``get_name`` otherwise.

  >>> for bcdata in PT.get_nodes_from_predicates(node, 'BCDataSet_t/BCData_t'):
  >>>   # Do something wih bcdata nodes

Note that the generic versions :func:`get_..._from_predicates` expect a list
of predicate functions as condition, and can thus be used to mix the kind
of criteria used to compare at each level :

>>> is_bc = lambda n : PT.get_label(n) == 'BC_t' and 
...                    PT.get_node_from_label(n, 'BCDataSet_t') is None
>>> is_pr = lambda n : PT.get_name(n) == 'PointRange'
>>> for pr in PT.get_nodes_from_predicates(node, [is_bc, is_pr]):
>>>   # Do something with pr nodes




Fine tuning searches
^^^^^^^^^^^^^^^^^^^^

Here is a selection of the most usefull *kwargs* accepted by the functions.
See API reference for the full list.

- ``depth`` (integer): *Apply to all the functions*

  Restrict the search to the nth level of children, where level 0 is the input node itself.
  If unset, search is not restricted.

  .. tip:: Limiting the search to a **single** level is so common that all the functions
    have a specific variant to do that : 

    - :func:`get_child_from_...` means :func:`get_node_from_...` with ``depth==1``
    - :func:`get_children_from_...` means :func:`get_nodes_from_...` with ``depth==1``

  >>> PT.get_nodes_from_label(node, 'GridLocation_t')
  # Return the 3 GridLocation nodes
  >>> PT.get_nodes_from_label(node, 'GridLocation_t', depth=2)
  # Return the 2 GridLocation nodes under the BC nodes

- ``explore`` ('shallow' or 'deep'): *Apply to get_nodes_from_... functions* 

  If explore == 'shallow', the children of a node matching the predicate
  are not tested. If explore='deep', all the nodes are tested.
  Default is 'shallow'.

  >>> PT.get_nodes_from_label(node, 'BC*_t')
  # Return nodes BC1 and BC2
  >>> PT.get_nodes_from_label(node, 'BC*_t', explore='deep')
  # Return nodes BC1, BCDataSet and BC2

- ``ancestors`` (bool): *Advanced -- Apply to get_..._from_predicates functions* 

  If ``True``, return tuple of nodes instead of the terminal node. Tuple is of size 
  ``len(conditions)`` and contains all the intermediate results. Default is ``False``.

  >>> for bc, loc in PT.get_nodes_from_predicates(node, 
  ...                                             'BC_t/GridLocation_t',
  ...                                             ancestors=True):
  ...   print(PT.get_name(bc), PT.get_value(loc))
  BC1 Vertex
  BC1 FaceCenter
  BC2 Vertex






Summary
-------

Here is an overview of the available searching functions:

.. table:: Generic version of search functions

  +---------------------+----------------------------------+-----------------------------------+-----------------------------------+
  |                     | Return first match               | Return all matches                | Iterate all matches               |
  +=====================+==================================+===================================+===================================+
  | Single predicate    | :func:`get_node_from_predicate`  | :func:`get_nodes_from_predicate`  | :func:`iter_nodes_from_predicate` |
  +---------------------+----------------------------------+-----------------------------------+-----------------------------------+
  | Multiple predicates | :func:`get_node_from_predicates` | :func:`get_nodes_from_predicates` | :func:`iter_nodes_from_predicates`|
  +---------------------+----------------------------------+-----------------------------------+-----------------------------------+

.. table:: Specialized versions of search functions
  
  +---------------------+-----------------------------------------+------------------------------------------+-------------------------------------------+
  |                     | Return first match                      | Return all matches                       | Iterate all matches                       |
  +=====================+=========================================+==========================================+===========================================+
  | Single predicate    | :func:`get_node_from_name`              | :func:`get_nodes_from_name`              | :func:`iter_nodes_from_name`              |
  |                     | :func:`get_node_from_label`             | :func:`get_nodes_from_label`             | :func:`iter_nodes_from_label`             |
  |                     | :func:`get_node_from_value`             | :func:`get_nodes_from_value`             | :func:`iter_nodes_from_value`             |
  |                     | :func:`get_node_from_name_and_label`    | :func:`get_nodes_from_name_and_label`    | :func:`iter_nodes_from_name_and_label`    |
  |                     | :func:`get_child_from_name`             | :func:`get_children_from_name`           | :func:`iter_children_from_name`           |
  |                     | :func:`get_child_from_label`            | :func:`get_children_from_label`          | :func:`iter_children_from_label`          |
  |                     | :func:`get_child_from_value`            | :func:`get_children_from_value`          | :func:`iter_children_from_value`          |
  |                     | :func:`get_child_from_name_and_label`   | :func:`get_children_from_name_and_label` | :func:`iter_children_from_name_and_label` |
  +---------------------+-----------------------------------------+------------------------------------------+-------------------------------------------+
  | Multiple predicates | :func:`get_node_from_names`             | :func:`get_nodes_from_names`             | :func:`iter_nodes_from_names`             |
  |                     | :func:`get_node_from_labels`            | :func:`get_nodes_from_labels`            | :func:`iter_nodes_from_labels`            |
  |                     | :func:`get_node_from_values`            | :func:`get_nodes_from_values`            | :func:`iter_nodes_from_values`            |
  |                     | :func:`get_node_from_name_and_labels`   | :func:`get_nodes_from_name_and_labels`   | :func:`iter_nodes_from_name_and_labels`   |
  |                     | :func:`get_child_from_names`            | :func:`get_children_from_names`          | :func:`iter_children_from_names`          |
  |                     | :func:`get_child_from_labels`           | :func:`get_children_from_labels`         | :func:`iter_children_from_labels`         |
  |                     | :func:`get_child_from_values`           | :func:`get_children_from_values`         | :func:`iter_children_from_values`         |
  |                     | :func:`get_child_from_name_and_labels`  | :func:`get_children_from_name_and_labels`| :func:`iter_children_from_name_and_labels`|
  +---------------------+-----------------------------------------+------------------------------------------+-------------------------------------------+

The following functions do not directly derive from the previous one,
but allow additional usefull searches:

.. autosummary::
  ~maia.pytree.get_node_from_path
  ~maia.pytree.get_all_CGNSBase_t
  ~maia.pytree.get_all_Zone_t


API reference
-------------

.. autofunction:: maia.pytree.get_node_from_predicate
.. autofunction:: maia.pytree.get_nodes_from_predicate
.. autofunction:: maia.pytree.iter_nodes_from_predicate
.. autofunction:: maia.pytree.get_node_from_predicates
.. autofunction:: maia.pytree.get_nodes_from_predicates
.. autofunction:: maia.pytree.iter_nodes_from_predicates

.. autofunction:: maia.pytree.get_node_from_path
.. autofunction:: maia.pytree.get_all_CGNSBase_t
.. autofunction:: maia.pytree.get_all_Zone_t



.. 
  .. autofunction:: maia.pytree.get_node_from_name
  .. autofunction:: maia.pytree.get_node_from_label
  .. autofunction:: maia.pytree.get_node_from_value
  .. autofunction:: maia.pytree.get_node_from_name_and_label
  .. autofunction:: maia.pytree.get_child_from_name
  .. autofunction:: maia.pytree.get_child_from_label
  .. autofunction:: maia.pytree.get_child_from_value
  .. autofunction:: maia.pytree.get_child_from_name_and_label

  .. autofunction:: maia.pytree.get_nodes_from_name
  .. autofunction:: maia.pytree.get_nodes_from_label
  .. autofunction:: maia.pytree.get_nodes_from_value
  .. autofunction:: maia.pytree.get_nodes_from_name_and_label
  .. autofunction:: maia.pytree.get_children_from_name
  .. autofunction:: maia.pytree.get_children_from_label
  .. autofunction:: maia.pytree.get_children_from_value
  .. autofunction:: maia.pytree.get_children_from_name_and_label
  .. autofunction:: maia.pytree.iter_nodes_from_name
  .. autofunction:: maia.pytree.iter_nodes_from_label
  .. autofunction:: maia.pytree.iter_nodes_from_value
  .. autofunction:: maia.pytree.iter_nodes_from_name_and_label
  .. autofunction:: maia.pytree.iter_children_from_name
  .. autofunction:: maia.pytree.iter_children_from_label
  .. autofunction:: maia.pytree.iter_children_from_value
  .. autofunction:: maia.pytree.iter_children_from_name_and_label

  .. autofunction:: maia.pytree.get_node_from_names
  .. autofunction:: maia.pytree.get_node_from_labels
  .. autofunction:: maia.pytree.get_node_from_values
  .. autofunction:: maia.pytree.get_node_from_name_and_labels
  .. autofunction:: maia.pytree.get_child_from_names
  .. autofunction:: maia.pytree.get_child_from_labels
  .. autofunction:: maia.pytree.get_child_from_values
  .. autofunction:: maia.pytree.get_child_from_name_and_labels

  .. autofunction:: maia.pytree.get_nodes_from_names
  .. autofunction:: maia.pytree.get_nodes_from_labels
  .. autofunction:: maia.pytree.get_nodes_from_values
  .. autofunction:: maia.pytree.get_nodes_from_name_and_labels
  .. autofunction:: maia.pytree.get_children_from_names
  .. autofunction:: maia.pytree.get_children_from_labels
  .. autofunction:: maia.pytree.get_children_from_values
  .. autofunction:: maia.pytree.get_children_from_name_and_labels
  .. autofunction:: maia.pytree.iter_nodes_from_names
  .. autofunction:: maia.pytree.iter_nodes_from_labels
  .. autofunction:: maia.pytree.iter_nodes_from_values
  .. autofunction:: maia.pytree.iter_nodes_from_name_and_labels
  .. autofunction:: maia.pytree.iter_children_from_names
  .. autofunction:: maia.pytree.iter_children_from_labels
  .. autofunction:: maia.pytree.iter_children_from_values
  .. autofunction:: maia.pytree.iter_children_from_name_and_labels

