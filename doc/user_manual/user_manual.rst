.. _user_manual:

###########
User Manual
###########

Maia methods are accessible through three main modules : 

- **Factory** module allows to generate maia trees, either from an other kind of tree
  (eg: the partitioning operation) or directly in memory. Factory functions return a new tree
  whose nature generally differs from the input tree.

- **Algo** module is the main maia module and provides parallel algorithms to be applied on maia trees.
  Some algorithm are only available for distributed trees (algo.dist) and some are only
  available for partitioned trees (algo.part). Few algorithm have an implementation for both kind
  of trees and are thus directly accessible through the algo module.
  
  Algo functions either modify their input tree or return some data, but they do not change the nature
  of the tree.

- **Transfer** module is a small module allowing to exchange data between maia trees.

Using maia trees in your application often consists in chaining functions from this different modules.
A typical workflow could be:

1. Load a structured tree from a file, which produce a **dist tree**.
2. Apply some distributed algorithms to this tree: for example a structured to unstructured conversion (algo.dist module).
3. Generate a corresponding partitionned tree (factory module).
4. Apply some partitioned algorithms to the **part tree**, such as wall distance computation (algo.part module),
   and even call you own tools (probably a CFD solver)
5. Transfer the resulting fields to the **dist tree** (transfer module).
6. Save the updated dist tree to disk.
 
This user manuel describes the main functions available in each module.

.. toctree::
  :maxdepth: 1

  factory
  algo
  transfer

.. std_elements_to_ngons
  
