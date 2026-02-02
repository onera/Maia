Pytree extension
================

Maia provides an extension to :ref:`the pytree module<pytree_module>` in order to
gather the application specific tree manipulation shortcuts: ``pytree.maia``.
Unless otherwise specified, all the functions described here are accessible
from the module namespace, shortened as ``MT``::

  import maia.pytree.maia as MT

Node inspection
---------------

Extends :ref:`pt_inspect`.


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


Tree operations
---------------
