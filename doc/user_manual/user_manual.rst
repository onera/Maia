.. _user_manual:

###########
User Manual
###########

Maia methods are accessible through three main modules : 

- **Factory** allows to generate Maia trees, generally from another kind of tree
  (e.g. the partitioning operation). Factory functions return a new tree
  whose nature generally differs from the input tree.

- **Algo** is the main Maia module and provides parallel algorithms to be applied on Maia trees.
  Some algorithms are only available for :ref:`distributed trees <user_man_dist_algo>`
  and some are only available for :ref:`partitioned trees <user_man_part_algo>`.
  A few algorithms are implemented for both kind
  of trees and are thus directly accessible through the  :ref:`algo <user_man_gen_algo>` module.
  
  Algo functions either modify their input tree inplace, or return some data, but they do not change the nature
  of the tree.

- **Transfer** is a small module allowing to transfer data between Maia trees. A transfer function operates
  on two existing trees and enriches the destination tree with data fields of the source tree.

Using Maia trees in your application often consists in chaining functions from these different modules.

.. image:: ./images/workflow.svg

A typical workflow could be:

1. Load a structured tree from a file, which produces a **dist tree**.
2. Apply some distributed algorithms to this tree: for example a structured to unstructured conversion (``algo.dist`` module).
3. Generate a corresponding partitionned tree (``factory`` module).
4. Apply some partitioned algorithms to the **part tree**, such as wall distance computation (``algo.part`` module),
   and even call you own tools (e.g. a CFD solver)
5. Transfer the resulting fields to the **dist tree** (``transfer`` module).
6. Save the updated dist tree to disk.
 
This user manuel describes the main functions available in each module.

.. toctree::
  :maxdepth: 1
  :hidden:

  factory
  algo
  transfer

.. std_elements_to_ngons
  
