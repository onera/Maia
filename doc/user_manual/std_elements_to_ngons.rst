
.. _std_elements_to_ngons:

Transform an mesh by elements to a polygonal mesh
=================================================


.. code-block:: python

  maia.std_elements_to_ngons(dist_tree_elts, comm)


Take a distributed `CGNSTree_t` or `CGNSBase_t`, and transform it into a NGon/NFace mesh. The tree is modified in-place.

Example
-------

Arguments
---------

:code:`dist_tree_elts`
  a :ref:`distributed tree <distributed_trees>` with
    * unstructured zones that contain regular element sections (i.e. not NGON, not NFACE and not MIXED)
    * the :ref:`Maia tree <maia_trees>` property

:code:`comm`
  a communicator over which `std_elements_to_ngons` is called

Output
------

The tree is modified in-place. The regular element sections are replaced by:
* a NGon section with:
  * An code:`ElementStartOffset`/code:`ElementConnectivity` node describing:
    * first the external faces in exactly the same order as they were (in particular, gathered by face type)
    * then the internal faces, also gathered by face type
  * a code:`ParentElements` node and a code:`ParentElementsPosition` node
* a NFace section with cells ordered as they were, with ids only shifted by the number of interior faces


Parallelism
-----------

`std_elements_to_ngons` is a collective function.

Complexity
----------

With :math:`N` the number of zones, :math:`n_i` the number of elements of zone :math:`i`, :math:`nf_i` its number of interior faces, and :math:`K` the number of processes

Sequential time: :math:`O(\sum_{i} n_i log(n_i))` (sort all faces)

Parallel time: :math:`O(\sum_{i} n_i/K log(n_i/K))`.

Theoretical scaling: :math:`1 - log(K)/log(n_i)`
Note: the scaling is computed as :math:`tp / ts * K` where :math:`ts` is the sequential time and :math:`tp` the parallel time. A scaling of 1 is perfect.
(Experimentally, the scaling seems poor - under investigation)


Peak memory: Approximately the size of the input tree + the output tree, i.e. :math:`\sum_{i} 2*n_i + nf_i` (:math:`*n_i` counted twice: once for the input element connectivity, once for the output NFace connectivity)

Size of communications: Approximately :math:`\sum_{i} 3 nf_i + n_i` all_to_all calls (for each zone, one call to sort interior faces, one to send back the face to the NFace, one to concatenate all faces, one to concatenate all cells)

Number of communication calls: Should be :math:`O(\sum_{i} log(n_i/K))` (number of iterations to find a balanced distribution of interior faces)

Note: in practice, :math:`nf_i` varies from :math:`2 n_i` (tet-dominant meshes) to :math:`3 n_i` (hex-dominant meshes).
