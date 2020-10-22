#pragma once 

/* Maillage "one_quad"
Le maillage utilisé est le suivant:
           _      y      j
         /_/|     |_ x   |_ i
        |_|/     /      /
                z      k     
Les ids des noeuds sont les suivants: (ordre fortran, indexé à 0):
           ________________
          /2               /3
         /                /|
        /__|____________ / |
       6|              7|  |
        |  |            |  |
        |               |  |
        |  |            |  |
        |   _ _ _ _ _ _ | _|
        |   0           |  /1
        | /             | /
        |_______________|/
        4               5

Le noeud 0 est en (3.,0.,0.) et le côté du cube est 1.
*/

/* Maillage "six_quads"
Le maillage utilisé est le suivant:
           _ _ _
         /_/_/_/|     y      j
        |_|_|_|/|     |_ x   |_ i
        |_|_|_|/     /      /
                    z      k

Les ids des noeuds sont les suivants (ordre fortran, indexé à 0):
           ________________________________________________
          /8              /9              /10              /11
         /               /               /                /|
        /__|____________/__|____________/__|____________ / |
      20|             21|             22|             23|  |
        |  |            |  |            |  |            |  |
        |               |               |               |  |
        |  |            |  |            |  |            |  |
        |   _ _ _ _ _ _ |   _ _ _ _ _ _ |   _ _ _ _ _ _ | _|
        |   4           |   5           |   6           |  /7
        | /             | /             | /             | /|
        |__|____________|__|____________|__|____________|/ |
      16|             17|             18|             19|  |
        |  |            |  |            |  |            |  |
        |               |               |               |  |
        |  |            |  |            |  |            |  |
        |   _ _ _ _ _ _ |   _ _ _ _ _ _ |   _ _ _ _ _ _ | _|
        |   0           |   1           |   2           |  /3
        | /             | /             | /             | /
        |_______________|_______________|_______________|/
       12              13              14               15

Le noeud 0 est en (0.,0.,0.) et le côté de chaque cube est 1.
*/
