.. _std_elements_to_ngons:

std_elements_to_ngons
=====================

Description
-----------

.. code-block:: python

  maia.transform.std_elements_to_ngons(dist_tree_elts, comm)


Take a **distributed** :cgns:`CGNSTree_t` or :cgns:`CGNSBase_t`, and transform it into a **distributed** :cgns:`NGon/NFace` mesh. The tree is modified in-place.

Example
-------

.. code-block:: python

  import maia.io
  import maia.transform
  from mpi4py import MPI
  comm = MPI.COMM_WORLD

  dist_tree = maia.io.file_to_dist_tree('element_mesh.cgns', comm)
  maia.transform.std_elements_to_ngons(dist_tree, comm)
  maia.io.dist_tree_to_file(dist_tree, 'poly_mesh.cgns', comm)

Arguments
---------

:code:`dist_tree_elts`
  a :ref:`distributed tree <dist_tree>` with
    * unstructured zones that contain regular element sections (i.e. not NGON, not NFACE and not MIXED)
    * the :ref:`Maia tree <maia_tree>` property

:code:`comm`
  a communicator over which :code:`std_elements_to_ngons` is called

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

:code:`std_elements_to_ngons` is a collective function.

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

Design alternatives
-------------------

The final step of the computation involves concatenating faces and cells global section by global section. This requires two heavyweight **all_to_all** calls. An alternative would be to concatenate locally. This would imply two trade-offs:

* the faces and cells would then not be globally gathered by type, and the exterior faces would not be first
* all the :cgns:`PointLists` (including those where :cgns:`GridLocation=FaceCenter`) would have to be shifted
