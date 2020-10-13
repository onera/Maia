#TODO

## Naming
Filter -> i/o (why call it filter? no biblio element to support the use, filter is too general and does not warn we are doing i/o)

## Licensing
MPL2?

## CMake
clean UseCython, MixPythonCythonModule, TestCreate
copy of python files to build is broken: does not remove files removed from source dir (solution: use the source dir directly?)

## Algorithms
load_collective_size_tree: replace by tree traversal + pruning predicate
attach_dims_to_dist_tree: fix convertFile2PyTree + skeletonData (skeletonData should be placed directly as #Size nodes)

uniform_distribution: where should we put it? (python std_e?)

## Tests
Unit tests: seq/par python/c++
Component, Perfo: seq/par python/c++
Functional (non-reg): seq/par par fichier / python
Deployment: gitlab+yaml+bash

Tests fonctionnels
n_tests_func, weight per test, n_core per test
n_cores_env
Exécuter tous les tests avec charge optimale (parallèle dans parallèle)

Algo :
1) tri par poids * n_core (plus grand d'abord)
2) tout le monde voit les tests
3) 

50 30 21 15 15 10 4 3 1 1 1 1 1

50 30 15 4 1
21 15 10 3 1 1 1 1

50 30 15 4
21 15 10 3

for i_test in sort_list:
  sub_comm_test = i_test.sub_comm
  if sub_comm_test==MPI_COMM_NULL:
    continue
  else:
    call pytest(sub_comm_test)

50
30
15
4 et 1
~~~~~~
15

