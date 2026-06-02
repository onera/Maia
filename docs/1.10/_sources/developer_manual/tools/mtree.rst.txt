.. _maia_pytree_module:

Pytree extension
================

Maia provides an extension to :ref:`the pytree module<pytree_module>` in order to
gather the application specific tree manipulation shortcuts: ``pytree.maia``.
Unless otherwise specified, all the functions described here are accessible
from the module namespace, shortened as ``MT``::

  import maia.pytree.maia as MT

Node inspection
---------------

The following functions extend :ref:`pt_inspect` and provide getters for data
specific to maia trees.
They use the same namespace classification as pytree.
Within each namespace, the functions provided are typically getters for:

- the *local* sizes of a distributed (:func:`dn_xxx`) or partitioned (:func:`pn_xxx`) mesh
- the *global* size of a distributed or partitioned mesh (:func:`n_xxx`), collectively computed in the partitioned case
- the distribution (:func:`distribution`) or global numbering (:func:`globalnumbering`) arrays


.. autoclass:: maia.pytree.maia.Zone
  :members:

.. autoclass:: maia.pytree.maia.Element
  :members:

.. autoclass:: maia.pytree.maia.Subset
  :members:



Node creation presets
---------------------

The following functions extend :ref:`node creation presets<pt_presets>` to easily
create Maia-specific nodes.

.. hint:: Unlike :func:`PT.new_xxx` functions, these functions update the
  target node if already existing.

.. autofunction:: maia.pytree.maia.new_Distribution
.. autofunction:: maia.pytree.maia.new_GlobalNumbering


Node searching
--------------

The following functions extend :ref:`searches<pt_node_search>` to easily
get Maia-specific nodes.

.. autofunction:: maia.pytree.maia.get_Distribution
.. autofunction:: maia.pytree.maia.get_GlobalNumbering

.. seealso:: These two functions have counterpart :func:`find_Distribution` and :func:`find_GlobalNumbering` that
  raise an error if no result is found.

.. autofunction:: maia.pytree.maia.get_partitioned_zones

Tree operations
---------------

The following functions extend :ref:`pt_operations` with operations
running on parallel trees.

Tree editing
^^^^^^^^^^^^

.. autofunction:: maia.pytree.maia.rename_zones

Tree comparisons
^^^^^^^^^^^^^^^^

The ``MT`` extension provides comparison operators that can
be used by :func:`~maia.pytree.diff_tree` to compare **distributed**
trees.
These functions are available in ``MT.compare`` namespace.

.. autofunction:: maia.pytree.maia.compare.EqualArray
.. autofunction:: maia.pytree.maia.compare.FieldComparison
.. autofunction:: maia.pytree.maia.compare.TensorFieldComparison