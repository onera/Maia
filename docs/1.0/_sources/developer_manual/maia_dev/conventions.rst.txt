Conventions
===========

Naming conventions
------------------

* **snake_case**
* A variable holding a number of things is written :code:`n_thing`. Example: :code:`n_proc`, :code:`n_vtx`. The suffix is singular.
* For unit tests, when testing variable :code:`<var>`, the hard-coded expected variable is named :code:`expected_<var>`.
* Usual abbreviations

  * **elt** for **element**
  * **vtx** for **vertex**
  * **proc** for **process**
  * **sz** for **size** (only for local variable names, not functions)

* Connectivities

  * **cell_vtx** means mesh array of cells described by their vertices (CGNS example: :cgns:`HEXA_8`)
  * **cell_face** means the cells described by their faces (CGNS example: :cgns:`NFACE_n`)
  * **face_cell** means for each face, the two parent cells (CGNS example: :cgns:`ParentElement`)
  * ... so on: **face_vtx**, **edge_vtx**...
  * **elt_vtx**, **elt_face**...: in this case, **elt** can be either a cell, a face or an edge


Other conventions
-----------------

We try to use semi-open intervals and 0-indexed structures. This is coherent with Python, C++, and their libraries, but unfortunately not with CGNS.
