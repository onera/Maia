Conventions
===========

Naming conventions
------------------

* A variable holding a number of things is written :code:`n_thing`. Example: :code:`n_proc`, :code:`n_vtx`. The suffix is singular.
* For unit tests, when testing variable :code:`<var>`, the hard-coded expected variable is named :code:`expected_<var>`.
* Usual abbreviations
  * **elt** for **element**
  * **vtx** for **vertex**
  * **proc** for **process**
  * **sz** for **size** (only for local variable names, not functions)


Other conventions
-----------------

We try to use semi-open intervals and 0-indexed structures. This is coherent with Python, C++, and their libraries, but unfortunately not with CGNS.
