.. _pt_yaml_parsing:

Yaml parsing
============

``maia.pytree`` does not come with any tool to read or write CGNS databases
from ADF or HDF files.
This choice has been made in order to maintain a light and portable library,
assuming that users will rely on an external software (such as ``maia``) to
perfom IO operations.

Nevertheless, for developement and debug purpose, we provide a converter mapping
CGNS trees to YAML files.
This is especially useful to prepare sample trees for unit testing in convenient 
and visual way.

Format description
------------------

To illustrate how the yaml format is used to store CGNSTrees, let's consider this
example file:

.. code-block:: yaml
  :linenos:

  Base Base_t [2,2]:
    Zone Zone_t [[9,4,0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0, 0.5, 1,  0,   0.5, 1.,  0, 0.5, 1]:
        CoordinateY DataArray_t R8 [0, 0,   0,  0.5, 0.5, 0.5, 1, 1,   1]:
      QUAD Elements_t [7, 0]: # This is a comment
        ElementRange IndexRange_t [1, 4]:
        ElementConnectivity DataArray_t:
          I4 : [1,2,5,4,  2,3,6,5,  
                4,5,8,7,  5,6,9,8]

The following rules must be observed to have a valid description of a CGNSTree:

**Comments**: short line comment can be used thanks to the ``#`` character *(l7)*.

**Hierarchic structure**: indentation is used to indicate that a node is the child of another node *(l3)*.  

**Node definition**: a node is defined by the pattern ``{name} {label} {value_kind} {value}:``.
Note the importance of the ``:``, which acts as a end-of-line marker.
The value, as well as its kind value_kind, are optional *(l2, l4)*.

**Values**: if a ``value`` is provided, it can be a string or a sequence of numbers *(l3, l1)*.
Nested sequences can be used to describe dimensional arrays *(l2)*. ``value`` can be ommited
if the node has no value *(l4)*.

**Values kind** : if ``value_kind`` is not provided, values are converted using
:func:`~maia.pytree.set_value` rules *(l2)*.
Otherwise, ``value_kind`` should be one of the CGNS value identifiers (eg. ``I4``, ``R8``, etc.) and will
be used to enforce the requested kind *(l5)*. ``value_kind`` can not exists if ``value`` is not provided.

**Long node definition**: alternatively, nodes can be defined by the pattern shown on lines 9-11: this
is especially useful when values are long arrays. When this pattern is used, ``value_kind`` is mandatory.
Note the additional indentation for the value definition and the lack of ``:`` at the end of the value sequence
(this is because end of value is now detected throught the indentation level).


YAML-CGNS converter
-------------------

.. attention:: The functions documented here are exposed in the ``maia.pytree.yaml`` module.

.. rubric:: From YAML to CGNS

.. autofunction:: maia.pytree.yaml.parse_yaml_cgns.to_node
.. autofunction:: maia.pytree.yaml.parse_yaml_cgns.to_nodes
.. autofunction:: maia.pytree.yaml.parse_yaml_cgns.to_cgns_tree

.. rubric:: From CGNS to YAML

.. autofunction:: maia.pytree.yaml.parse_cgns_yaml.to_yaml