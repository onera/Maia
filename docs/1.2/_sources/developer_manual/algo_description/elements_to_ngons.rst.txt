.. _elements_to_ngons_impl:

elements_to_ngons
=================

Description
-----------

.. code-block:: python

  maia.algo.dist.elements_to_ngons(dist_tree_elts, comm)

Take a **distributed** :cgns:`CGNSTree_t` or :cgns:`CGNSBase_t`, and transform it into a **distributed** :cgns:`NGon/NFace` mesh. The tree is modified in-place.

Example
-------

.. literalinclude:: ../../user_manual/snippets/test_algo.py
  :start-after: #elements_to_ngons@start
  :end-before: #elements_to_ngons@end
  :dedent: 2

Arguments
---------

:code:`dist_tree_elts`
  a :ref:`distributed tree <dist_tree>` with
    * unstructured zones that contain regular element sections (i.e. not NGON, not NFACE and not MIXED)
    * the :ref:`Maia tree <maia_tree>` property

:code:`comm`
  a communicator over which :code:`elements_to_ngons` is called

Output
------

The tree is modified in-place. The regular element sections are replaced by:

* a NGon section with:

  * An :cgns:`ElementStartOffset`/:cgns:`ElementConnectivity` node describing:

    * first the external faces in exactly the same order as they were (in particular, gathered by face type)
    * then the internal faces, also gathered by face type

  * a :cgns:`ParentElements` node and a :cgns:`ParentElementsPosition` node

* a NFace section with cells ordered as they were, with ids only shifted by the number of interior faces


Parallelism
-----------

:code:`elements_to_ngons` is a collective function.

Complexity
----------

With :math:`N` the number of zones, :math:`n_i` the number of elements of zone :math:`i`, :math:`n_{f,i}` its number of interior faces, and :math:`K` the number of processes

Sequential time
  :math:`\displaystyle \mathcal{O} \left( \sum_{i=0}^N n_i \cdot log(n_i) \right)`

  The cost is dominated by the sorting of faces (interior and exterior) used to spot identical ones.

Parallel time
  :math:`\displaystyle \mathcal{O} \left( \sum_{i=0}^N n_i/K \cdot log(n_i) \right)`

  The parallel distributed sort algorithm consists of three steps:

    1. A partitioning step that locally gather faces into :math:`K` buckets that are of equal global size. This uses a :math:`K`-generalized variant of `quickselect <https://fr.wikipedia.org/wiki/Quickselect>`_ that is of complexity :math:`\displaystyle \mathcal{O} \left( n_i/K \cdot log(K) \right)`
    2. An all_to_all exchange step that gather the :math:`K` buckets on the :math:`K` processes. This step is not accounted for here (see below)
    3. A local sorting step that is :math:`\displaystyle \mathcal{O} \left( n_i/K \cdot log(n_i/K) \right)`

  If we sum up step 1 and 3, we get

.. math::

  \begin{equation} \label{eq1}
  \begin{split}
     \textrm{C}_\textrm{parallel_sort} & = \mathcal{O} \left( n_i/K \cdot log(K) + n_i/K \cdot log(n_i/K) \right) \\
                                       & = \mathcal{O} \left( n_i/K \cdot \left( log(K) + log(n_i/K) \right) \right) \\
                                       & = \mathcal{O} \left( n_i/K \cdot log(n_i) \right)
  \end{split}
  \end{equation}

Theoretical scaling
  :math:`\textrm{Speedup} = K`

  Experimentally, the scaling is much worse - under investigation.

  Note: the speedup is computed by :math:`\textrm{Speedup} = t_s / t_p` where :math:`t_s` is the sequential time and :math:`t_p` the parallel time. A speedup of :math:`K` is perfect, a speedup lower than :math:`1` means that sequential execution is faster.

Peak memory
  Approximately :math:`\displaystyle \sum_{i=0}^N 2 n_i + n_{f,i}`

  This is the size of the input tree + the output tree. :math:`n_i` is counted twice: once for the input element connectivity, once for the output NFace connectivity

Size of communications
  Approximately :math:`\displaystyle \sum_{i=0}^N 3 n_{f,i} + n_i` **all_to_all** calls

  For each zone, one **all_to_all** call to sort interior faces, one to send back the faces to the NFace, one to concatenate all faces, one to concatenate all cells

Number of communication calls
  Should be :math:`\displaystyle \mathcal{O} \left( \sum_{i=0}^N log(n_i/K) \right)`

  The number of communications is constant, except for the algorithm finding a balanced distribution of interior faces

Note
  In practice, :math:`n_{f,i}` varies from :math:`2 n_i` (tet-dominant meshes) to :math:`3 n_i` (hex-dominant meshes).

Algorithm explanation
---------------------

.. code-block:: c++

  maia::elements_to_ngons(cgns::tree& dist_zone, MPI_Comm comm)

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

Design alternatives
-------------------

The final step of the computation involves concatenating faces and cells global section by global section. This requires two heavyweight **all_to_all** calls. An alternative would be to concatenate locally. This would imply two trade-offs:

* the faces and cells would then not be globally gathered by type, and the exterior faces would not be first
* all the :cgns:`PointLists` (including those where :cgns:`GridLocation=FaceCenter`) would have to be shifted
