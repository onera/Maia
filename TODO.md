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
