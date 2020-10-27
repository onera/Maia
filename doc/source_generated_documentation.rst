.. _source_generated_documentation:

##############################
Source-generated documentation
##############################

TODO

.. .. doxygenindex::
..   :project: std_e


.. _data_structure:

.. currentmodule:: std_e

Heterogenous vector
===================

This structure allows to store in a tuple multiple type but ...


.. literalinclude::  /../std_e/data_structure/test/heterogenous_vector.test.cpp
  :start-after: [Sphinx Doc] hvector {
  :end-before: [Sphinx Doc] hvector }

This one is define here : :cpp:class:`std_e::hvector`

.. doxygenclass:: std_e::hvector
   :project: std_e
   :protected-members:
   :private-members:
   :members:

Algorithm on hvector
--------------------

for_each_element
^^^^^^^^^^^^^^^^

.. doxygenfunction:: std_e::for_each_element(hvector< Ts... > &hv, F f) -> void
   :project: std_e

.. doxygenfunction:: std_e::for_each_element(const hvector< Ts... > &hv, F f) -> void
   :project: std_e


find_apply
^^^^^^^^^^^

.. doxygenfunction:: std_e::find_apply(hvector< Ts... > &hv, Unary_pred p, F f) -> std::pair< int, int >
   :project: std_e


for_each_if
^^^^^^^^^^^

.. doxygenfunction:: std_e::for_each_if(hvector< Ts... > &hv, Unary_pred p, F f) -> void
   :project: std_e

Accessor on hvector
--------------------

.. doxygenfunction:: std_e::get(hvector< Ts... > &x) -> std::vector< T > &
   :project: std_e



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
