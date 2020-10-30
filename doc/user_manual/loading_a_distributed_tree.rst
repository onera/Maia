.. contents:: :local:

.. _load_dist_tree:

Loading a distributed tree
==========================

A *dist tree* is loaded in 3 steps :
1. Collectively load a *size tree*
2. Deduce a distribution over the processes from the sizes
3. Create a **hdf selector**
4. Load the *dist tree* in parallel


Filters
-------

fichier contient dataspace
noeud CGNS <-> dataspace hdf
citer doc hdf
infos:
https://support.hdfgroup.org/HDF5/Tutor/selectsimple.html
https://support.hdfgroup.org/HDF5/doc1.6/UG/12_Dataspaces.html
  début, taille de block, taille pour chaque block, stride
DSMMRYDA  offset, stride, size, block (hdf)
DSFILEDA  offset, stride, size, block (hdf)
DSGLOBDA  taille totale
DSFORMDA  0: contigu, 1: interlacé, 2: concaténé interlacé
