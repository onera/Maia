.. contents:: :local:

.. _load_dist_tree:

Loading a distributed tree
==========================

A *dist tree* is loaded in 3 steps :
1. collectively load a *size tree*
2. a distribution over the processes is deduced from the sizes
3. load the *dist tree* in parallel.


