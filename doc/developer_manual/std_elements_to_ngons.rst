
.. _std_elements_to_ngons_dev:

std_elements_to_ngons
=====================

See the :ref:`associated user manual entry <std_elements_to_ngons>`

.. code-block:: c++

  maia::std_elements_to_ngons(cgns::tree& dist_zone, MPI_Comm comm)


Take a distributed :cgns:`Zone_t` and transform it into a :cgns:`NGon/NFace` zone. The tree is modified in-place.

Algorithm explanation
---------------------

The algorithm is divided in two steps:

1. Generate interior faces, insert them between exterior faces and cells. During the process, also compute the parent information (:cgns:`ParentElements` and :cgns:`ParentElementsPosition`) and add the :cgns:`CellFace` connectivity to cells. All elements keep their standard element types, in particular interior faces are inserted by section type (e.g. there will be a :cgns:`TRI_3_interior` node and a :cgns:`QUAD_4_interior` node, not a :cgns:`NGon` node)
2. Transform all sections into :cgns:`NGon/NFace`

Generate interior faces
^^^^^^^^^^^^^^^^^^^^^^^

Simplified algorithm
""""""""""""""""""""

Let us look at a simplified sequential algorithm first:

.. code-block:: text

  1. For all element sections (3D and 2D):
    Generate faces
      => FaceVtx arrays (one for Tris, one for Quads)
      => Associated Parent and ParentPosition arrays
          (only one parent by element at this stage)
    Interior faces are generated twice (by their two parent cells)
    Exterior faces are generated twice (by a parent cell and a boundary face)

  2. For each element kind in {Tri,Quad}
    Sort the FaceVtx array
      (ordering: lexicographical comparison of vertices)
    Sort the parent arrays accordingly
      => now each face appears consecutively exactly twice
          for interior faces,
            the FaceVtx is always inverted between the two occurences
          for exterior faces,
            it depends on the normal convention

  Note that interior and exterior faces can be distinguised
    by looking at the id of their parents

  3. For each interior face appearing twice:
    Create an interior face section where:
      the first FaceVtx is kept
      the two parents are stored
  4. For each exterior face appearing twice:
    One of the face is the original boundary section face,
      one was generated from the joint cell
    Send back to the original boundary face its parent face id and position
      => store the parent information of the boundary face

Parallel algorithm
""""""""""""""""""

The algorithm is very similar to the sequential one. We need to modify two operations:

Sorting of the FaceVtx array (step 2)
  The parallel sorting is done in three steps:

  1. apply a partial sort :cpp:`std_e::sort_by_rank` that will determine the rank of each FaceVtx
  2. call an :cpp:`all_to_all` communication step that sends each connectivity to its rank, based on the information of the previous step
  3. sort each received FaceVtx locally

Send back boundary parents and position to the original boundary faces (step 4)
  Since the original face can be remote, this is a parallel communication operation using :cpp:`std_e::scatter`

Computation of the CellFace
"""""""""""""""""""""""""""

After step 2, we have all the faces exactly once, with their parent ids and parent positions.
We can then compute the CellFace of each cell section by the following algorithm:

.. code-block:: text

  For each cell section:
    pre-allocate the CellFace array
      (its size is n_cell_in_section * n_face_of_cell_type)
    view it as a global distributed array
  For each unique face:
    For each of its parent cells (could be one or two):
      send the parent cell the id of the face and its position
      insert the result in the CellFace array

As previously, the send operation uses a **scatter** pattern

Transform all sections into NGon/NFace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Thanks to the previous algorithm, we have:

* all exterior and interior faces with their parent information
* the CellFace connectivity of the cell sections

Elements are ordered in something akin to this:

* boundary tris
* boundary quads
* internal tris
* internal quads

* tetras
* pyras
* prisms
* hexas

The algorithm then just needs to:

* concatenate all FaceVtx of the faces into a :cgns:`NGon` node and add a :cgns:`ElementStartOffset`
* concatenate all CellFace of the cells into a :cgns:`NFace` node and add a :cgns:`ElementStartOffset`
