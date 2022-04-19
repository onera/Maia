#pragma once

// TODO put this in doc
/* Maillage "one_quad"
Le maillage utilisé est le suivant:
           _      y      j
         /_/|     |_ x   |_ i
        |_|/     /      /
                z      k
Les ids des noeuds sont les suivants: (ordre fortran, indexé à 1):
           ________________
          /3               /4
         /                /|
        /__|____________ / |
       7|              8|  |
        |  |            |  |
        |               |  |
        |  |            |  |
        |   _ _ _ _ _ _ | _|
        |   1           |  /2
        | /             | /
        |_______________|/
        5               6

Le noeud 1 est en (3.,0.,0.) et le côté du cube est 1.
*/

/* Maillage "six_quads"
Le maillage utilisé est le suivant:
           _ _ _
         /_/_/_/|     y      j
        |_|_|_|/|     |_ x   |_ i
        |_|_|_|/     /      /
                    z      k

Les ids des noeuds sont les suivants (ordre fortran, indexé à 1):
           ________________________________________________
          /9              /10              /11              /12
         /               /               /                /|
        /__|____________/__|____________/__|____________ / |
      21|             22|             23|             24|  |
        |  |            |  |            |  |            |  |
        |       /4/     |      /5/      |       /6/     |  |
        |  |            |  |            |  |            |  |
        |   _ _ _ _ _ _ |   _ _ _ _ _ _ |   _ _ _ _ _ _ | _|
        |   5           |   6           |   7           |  /8
        | /             | /             | /             | /|
        |__|____________|__|____________|__|____________|/ |
      17|             18|             19|             20|  |
        |  |            |  |            |  |            |  |
        |       /1/     |       /2/     |       /3/     |  |
        |  |            |  |            |  |            |  |
        |   _ _ _ _ _ _ |   _ _ _ _ _ _ |   _ _ _ _ _ _ | _|
        |   1           |   2           |   3           |  /4
        | /             | /             | /             | /
        |_______________|_______________|_______________|/
       13              14              15               16


faces:
i
    2 4 6 8
    1 3 5 7
j
    15 16 17
    12 13 14
     9 10 11
k
  back
    21 22 23
    18 19 20
  front
    27 28 29
    24 25 26


Le noeud 1 est en (0.,0.,0.) et le côté de chaque cube est 1.
*/


#include <array>
#include <vector>
#include "std_e/utils/concatenate.hpp"

namespace maia::six_hexa_mesh {

  const std::vector< std::array<int,8> > cell_vtx = {
    {1,2, 6, 5,13,14,18,17},
    {2,3, 7, 6,14,15,19,18},
    {3,4, 8, 7,15,16,20,19},
    {5,6,10, 9,17,18,22,21},
    {6,7,11,10,18,19,23,22},
    {7,8,12,11,19,20,24,23}
  };

  const std::vector< std::array<int,4> > i_face_vtx = {
    {1, 5,17,13},
    {5, 9,21,17},
    {2, 6,18,14},
    {6,10,22,18},
    {3, 7,19,15},
    {7,11,23,19},
    {4, 8,20,16},
    {8,12,24,20}
  };

  const std::vector< std::array<int,4> > j_face_vtx = {
    { 1,13,14, 2},
    { 2,14,15, 3},
    { 3,15,16, 4},
    { 5,17,18, 6},
    { 6,18,19, 7},
    { 7,19,20, 8},
    { 9,21,22,10},
    {10,22,23,11},
    {11,23,24,12}
  };

  const std::vector< std::array<int,4> > k_face_vtx = {
    { 1, 2, 6, 5},
    { 2, 3, 7, 6},
    { 3, 4, 8, 7},
    { 5, 6,10, 9},
    { 6, 7,11,10},
    { 7, 8,12,11},
    {13,14,18,17},
    {14,15,19,18},
    {15,16,20,19},
    {17,18,22,21},
    {18,19,23,22},
    {19,20,24,23}
  };

  const auto face_vtx = std_e::concatenate(i_face_vtx,j_face_vtx,k_face_vtx);

  const std::vector<int> l_parents = {
    0,0,  1,4,  2,5,  3,6,        // faces normal to i, by "face sheets" at i=1/2/3/4
    0,0,0,  1,2,3,  4,5,6,       // faces normal to j, by "face sheets" at j=1/2/3
    0,0,0,0,0,0,  1,2,3,4,5,6 // faces normal to k, by "face sheets" at k=1/2
  };
  const std::vector<int> r_parents = {
    1,4,  2,5,  3,6,  0,0,
    1,2,3,  4,5,6,  0,0,0,
    1,2,3,4,5,6,  0,0,0,0,0,0
  };
} // maia::six_hexa_mesh

