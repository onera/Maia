"""
These numbering functions map cells, faces or vertex of a structured mesh,
identified by their three indices (i,j,k), to an absolute 1d identifier.
Some conventions shared by all the functions:
- (i,j,k) and resulting indices start at 1
- n_cell / n_vtx is the number of cells and vertices in the structured block
- Functions should be call directly on numpy arrays for optimal performance,
  possibly using vectorisation

We adopt the following conventions for numbering :



3D MESHES
*********

Vertices and cells are ordered first increasing i index, then j, then k:

        
                +----+----+----+-----+               41---42---43--44--- 45       
               / 13 / 14 / 15 / 16 / |              /    /    /    /    / |       
              +----+----+----+----+  +             36---37---38--39---40  30
    j ^      /    /    /    /    / |/|            /    /    /    /    / |/|
k ^  /      +----+----+----+----+  + +           31---32---33---34--35 25 15      
  │ /       |  9 | 10 | 11 | 12 | /|8/           |    |    |    |    | /| /       
  └───> i   +----+----+----+----+  +             16---17---18--19---20  10        
            |  1 |  2 |  3 |  4 | /              |    |    |    |    | / 
            +----+----+----+----+                1----2----3----4----5

For Faces, IFaces are numbered first, then JFaces, then KFaces; within each block, we number
increasing i, then increasing j, then increasing k



                          +           +           +           +           +                                                          
                         /|          /|          /|          /|          /|                                             
  I Faces Numbering     / |         / |         / |         / |         / |                                                  
    (1 ... 20)         +  |        +  |        +  |        +  |        +  |                                         
                      /|16+       /|17+       /|18+       /|19+       /|20+                                  
                     / | /|      / | /|      / | /|      / | /|      / | /|
                    +  |/ |     +  |/ |     +  |/ |     +  |/ |     +  |/ |                               
                    |11+  |     |12+  |     |13+  |     |14+  |     |15+  |                            
 k  j ^             | /|  +     | /|  +     | /|  +     | /|  +     | /|10+                                                  
  ^  /              |/ |6/      |/ |7/      |/ |8/      |/ |9/      |/ | /                                           
  │ /               +  |/       +  |/       +  |/       +  |/       +  |/                          
  └───> i           |  +        |  +        |  +        |  +        |  +                          
                    |1/         |2/         |3/         |4/         |5/                          
                    |/          |/          |/          |/          |/                        
                    +           +           +           +           +                     
                   i=1         i=2         i=3         i=4         i=5

                                                            +----+----+----+----+
         J Faces Numbering                                 | 41 | 42 | 43 | 44 |
           (21 ... 44)                                     +----+----+----+----+
                                   +----+----+----+----+   | 29 | 30 | 31 | 32 |
                                   | 37 | 38 | 39 | 40 |   +----+----+----+----+
k  j ^                             +----+----+----+----+       Back plane (j=3)
 ^  /      +----+----+----+----+   | 25 | 26 | 27 | 28 |
 │ /       | 33 | 34 | 35 | 36 |   +----+----+----+----+
 └───> i   +----+----+----+----+     Middle plane (j=2)
           | 21 | 22 | 23 | 24 |
           +----+----+----+----+           
             Front plane (j=1)
                                                                            +----+----+----+----+
                                                                           / 65 / 66 / 67 / 68 /
            K Faces Numbering                                             +----+----+----+----+   
              (45 ... 68)                     +----+----+----+----+      / 61 / 62 / 63 / 64 /    
                                             / 57 / 58 / 59 / 60 /      +----+----+----+----+     
                                            +----+----+----+----+          Top plane (k=3)
k  j ^          +----+----+----+----+      / 53 / 54 / 55 / 56 /      
 ^  /          / 49 / 50 / 51 / 52 /      +----+----+----+----+       
 | /          +----+----+----+----+        Middle plane (k=2)
 └───> i     / 45 / 46 / 47 / 48 /
            +----+----+----+----+  
             Bottom plane (k=1)


2D MESHES
*********

Vertices and cells are ordered first increasing i index, then j:

               +----+----+----+----+----+      19---20---21--22---23---24    
               | 11 | 12 | 13 | 14 | 15 |      |    |    |    |    |    |    
j ^            +----+----+----+----+----+      13---14---15--16---17---18      
  │            |  6 |  7 |  8 |  9 | 10 |      |    |    |    |    |    |    
  └───> i      +----+----+----+----+----+      7----8----9---10---11---12
               |  1 |  2 |  3 |  4 |  5 |      |    |    |    |    |    |
               +----+----+----+----+----+      1----2----3----4----5----6

For Edges, IEdges are numbered first, then JEdges; within each block, we number
increasing i, then increasing j

               +----+----+----+----+----+      +-34-+-35-+-36-+-37-+-38-+    
               13   14   15  16   17   18      |    |    |    |    |    |    
j ^            +----+----+----+----+----+      +-29-+-30-+-31-+-32-+-33-+      
  │            7    8    9   10   11   12      |    |    |    |    |    |    
  └───> i      +----+----+----+----+----+      +-24-+-25-+-26-+-27-+-28-+
               1    2    3    4    5    6      |    |    |    |    |    |
               +----+----+----+----+----+      +-19-+-20-+-21-+-22-+-23-+

  """

import numpy as np
from maia.typing import List, Tuple, Union, ArrayLike, DTypeLike, NDArray, Sequence
from cmaia.utils import numbering as cnumbering

def ijk_to_index_from_loc(i: Union[int, ArrayLike], 
                          j: Union[int, ArrayLike],
                          k: Union[int, ArrayLike],
                          loc: str, n_vtx: List[int]) -> Union[int, ArrayLike]:
  """Dispatch ijk to index for 3D meshes, depending of grid location"""
  n_cell = tuple(k-1 for k in n_vtx)
  if loc == 'Vertex':
    return ijk_to_index(i,j,k, n_vtx)
  elif loc == 'CellCenter':
    return ijk_to_index(i,j,k, n_cell)
  elif loc == 'IFaceCenter':
    return ijk_to_faceiIndex(i,j,k,n_cell,n_vtx)
  elif loc == 'JFaceCenter':
    return ijk_to_facejIndex(i,j,k,n_cell,n_vtx)
  elif loc == 'KFaceCenter':
    return ijk_to_facekIndex(i,j,k,n_cell,n_vtx)
  raise ValueError(f"Unsupported location '{loc}'.")

def index_to_ijk_from_loc(idx: Union[int, ArrayLike], 
                          loc: str,
                          n_vtx: List[int]) -> Tuple[Union[int, ArrayLike],
                                                     Union[int, ArrayLike],
                                                     Union[int, ArrayLike]]:
  """Dispatch index to ijk for 3D meshes, depending of grid location"""
  n_cell = tuple(k-1 for k in n_vtx)
  if loc == 'Vertex':
    return index_to_ijk(idx, n_vtx)
  elif loc == 'CellCenter':
    return index_to_ijk(idx, n_cell)
  elif loc == 'IFaceCenter':
    return faceiIndex_to_ijk(idx, n_cell, n_vtx)
  elif loc == 'JFaceCenter':
    return facejIndex_to_ijk(idx, n_cell, n_vtx)
  elif loc == 'KFaceCenter':
    return facekIndex_to_ijk(idx, n_cell, n_vtx)
  raise ValueError(f"Unsupported location '{loc}'.")

def ij_to_index_from_loc(i: Union[int, ArrayLike],
                         j: Union[int, ArrayLike], 
                         loc: str, n_vtx: List[int]) -> Union[int, ArrayLike]:
  """Dispatch ij to index for 2D meshes, depending of grid location"""
  n_cell = tuple(k-1 for k in n_vtx)
  if loc == 'Vertex':
    return ij_to_index(i,j, n_vtx)
  elif loc == 'CellCenter':
    return ij_to_index(i,j, n_cell)
  if loc == 'IEdgeCenter':
    return ij_to_edgeiIndex(i,j,n_cell,n_vtx)
  elif loc == 'JEdgeCenter':
    return ij_to_edgejIndex(i,j,n_cell,n_vtx)
  raise ValueError(f"Unsupported location '{loc}'.")

def index_to_ij_from_loc(idx: Union[int, ArrayLike],
                         loc: str,
                         n_vtx: List[int]) -> Tuple[Union[int, ArrayLike],
                                                    Union[int, ArrayLike]]:
  """Dispatch index to ij for 2D meshes, depending of grid location"""
  n_cell = tuple(k-1 for k in n_vtx)
  if loc == 'Vertex':
    return index_to_ij(idx, n_vtx)
  elif loc == 'CellCenter':
    return index_to_ij(idx, n_cell)
  elif loc == 'IEdgeCenter':
    return edgeiIndex_to_ij(idx, n_cell, n_vtx)
  elif loc == 'JEdgeCenter':
    return edgejIndex_to_ij(idx, n_cell, n_vtx)
  raise ValueError(f"Unsupported location '{loc}'.")

## 3D funcs

def ijk_to_index(i: Union[int, ArrayLike], 
                 j: Union[int, ArrayLike],
                 k: Union[int, ArrayLike],
                 n_elmt: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J,K) -> Idx for cells or vertices """
  return i + (j-1)*n_elmt[0] + (k-1)*n_elmt[0]*n_elmt[1]

def index_to_ijk(idx: Union[int, ArrayLike], 
                 n_elmt: Union[List[int], 
                 Tuple[int, ...]]
                 ) -> Tuple[Union[int, ArrayLike], Union[int, ArrayLike], Union[int, ArrayLike]]:
  """ Idx -> (I,J,K) for cells or vertices """
  k = ((idx - 1) // (n_elmt[0]*n_elmt[1])) + 1
  j = (idx - (k-1)*(n_elmt[0]*n_elmt[1]) - 1) // n_elmt[0] + 1
  i = idx - (j-1)*n_elmt[0] - (k-1)*(n_elmt[0]*n_elmt[1])
  return i,j,k

def ijk_to_faceiIndex(i: Union[int, ArrayLike], 
                      j: Union[int, ArrayLike], 
                      k: Union[int, ArrayLike], 
                      n_cell: Union[List[int], Tuple[int, ...]], 
                      n_vtx: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J,K) -> Idx for I-normal faces """
  return i + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_cell[1]

def faceiIndex_to_ijk(
  idx: Union[int, ArrayLike], 
  n_cell: Union[List[int], Tuple[int, ...]],
  n_vtx: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike], Union[int, ArrayLike], Union[int, ArrayLike]]:
  """ Idx -> (I,J,K) for I-normal faces """
  k = ((idx - 1) // (n_vtx[0]*n_cell[1])) + 1
  j = (idx - (k-1)*(n_vtx[0]*n_cell[1]) - 1) // n_vtx[0] + 1
  i = idx - (j-1)*n_vtx[0] - (k-1)*(n_vtx[0]*n_cell[1])
  return i,j,k

def ijk_to_facejIndex(
  i: Union[int, ArrayLike], 
  j: Union[int, ArrayLike], 
  k: Union[int, ArrayLike], 
  n_cell: Union[List[int], Tuple[int, ...]], n_vtx: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J,K) -> Idx for J-normal faces """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  return i + (j-1)*n_cell[0] + (k-1)*n_vtx[1]*n_cell[0] + nbFacesi

def facejIndex_to_ijk(idx: Union[int, ArrayLike],
                      n_cell: Union[List[int], Tuple[int, ...]], 
                      n_vtx: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike],
                                                                         Union[int, ArrayLike],
                                                                         Union[int, ArrayLike]]:
  """ Idx -> (I,J,K) for J-normal faces """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  k = ((idx - 1 - nbFacesi) // (n_vtx[1]*n_cell[0])) + 1
  j = (idx - (k-1)*(n_vtx[1]*n_cell[0]) - 1 - nbFacesi) // n_cell[0] + 1
  i = idx - (j-1)*n_cell[0] - (k-1)*(n_vtx[1]*n_cell[0]) - nbFacesi
  return i,j,k

def ijk_to_facekIndex(i: Union[int, ArrayLike],
                      j: Union[int, ArrayLike],
                      k: Union[int, ArrayLike], 
                      n_cell: Union[List[int], Tuple[int, ...]], 
                      n_vtx: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J,K) -> Idx for K-normal faces """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  nbFacesj = n_vtx[1]*n_cell[0]*n_cell[2]
  return i + (j-1)*n_cell[0] + (k-1)*n_cell[0]*n_cell[1] + nbFacesi + nbFacesj

def facekIndex_to_ijk(idx: Union[int, ArrayLike], 
                      n_cell: Union[List[int],
                      Tuple[int, ...]],
                      n_vtx: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike], 
                                                                         Union[int, ArrayLike], 
                                                                         Union[int, ArrayLike]]:
  """ Idx -> (I,J,K) for K-normal faces """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  nbFacesj = n_vtx[1]*n_cell[0]*n_cell[2]
  k = ((idx - 1 - nbFacesi - nbFacesj) // (n_cell[0]*n_cell[1])) + 1
  j = (idx - (k-1)*(n_cell[0]*n_cell[1]) - 1 - nbFacesi - nbFacesj) // n_cell[0] + 1
  i = idx - (j-1)*n_cell[0] - (k-1)*(n_cell[0]*n_cell[1]) - nbFacesi - nbFacesj
  return i,j,k


## 2D funcs

def ij_to_index(i: Union[int, ArrayLike], 
                j: Union[int, ArrayLike], 
                n_elmt: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J) -> Idx for cells or vertices """
  return i + (j-1)*n_elmt[0]

def index_to_ij(idx: Union[int, ArrayLike],
                n_elmt: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike],
                                                                    Union[int, ArrayLike]]:
  """ Idx -> (I,J) for cells or vertices """
  j = (idx-1) // n_elmt[0] + 1
  i = idx - (j-1)*n_elmt[0]
  return i,j

def ij_to_edgeiIndex(i: Union[int, ArrayLike], 
                     j: Union[int, ArrayLike],
                     n_cell: Union[List[int], Tuple[int, ...]], 
                     n_vtx: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J) -> Idx for I-normal 2D edges """
  return i + (j-1)*n_vtx[0]

def edgeiIndex_to_ij(idx: Union[int, ArrayLike], 
                     n_cell: Union[List[int], Tuple[int, ...]],
                     n_vtx: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike],
                                                                        Union[int, ArrayLike]]:
  """ Idx -> (I,J) for I-normal 2D edges """
  j = (idx - 1) // n_vtx[0] + 1
  i = idx - (j-1)*n_vtx[0]
  return i,j

def ij_to_edgejIndex(i: Union[int, ArrayLike],
                     j: Union[int, ArrayLike], 
                     n_cell: Union[List[int], Tuple[int, ...]], 
                     n_vtx: Union[List[int], Tuple[int, ...]]) -> Union[int, ArrayLike]:
  """ (I,J) -> Idx for J-normal 2D edges """
  nbEdgei = n_vtx[0]*n_cell[1]
  return i + (j-1)*n_cell[0] + nbEdgei

def edgejIndex_to_ij(idx: Union[int, ArrayLike], 
                     n_cell: Union[List[int], 
                     Tuple[int, ...]],
                     n_vtx: Union[List[int], Tuple[int, ...]]) -> Tuple[Union[int, ArrayLike], 
                                                                        Union[int, ArrayLike]]:
  """ Idx -> (I,J) for J-normal 2D edges """
  nbEdgei = n_vtx[0]*n_cell[1]
  j = (idx - 1 - nbEdgei) // n_cell[0] + 1
  i = idx - (j-1)*n_cell[0] - nbEdgei
  return i,j


###############################################################################

###############################################################################
def ngon_dconnectivity_from_gnum(bounds: Sequence[int],
                                 n_cell: Sequence[int],
                                 dtype: DTypeLike,
                                 ) -> Tuple[NDArray, NDArray]:
  """
  Generate a distributed ngon connectivity between the indicated face gnum ids for
  a zone of a given size.
  Faces will be generated for global id between
    [begin; endI[ for i-normal faces   Examples :
    [endI; endJ[  for j-normal faces    * [100, 200, 300, 300] -> generate ifaces 100-200 and jface 200-300
    [endJ; endK[  for k-normal faces    * [300, 300, 300, 400] -> generate kfaces 300-400
  Size of dist zone must be given as the number of cells (size=3)
  """
  n_face_loc = bounds[3] - bounds[0]
  face_pe  = np.empty((n_face_loc, 2), order='F', dtype=dtype)
  face_vtx = np.empty(4*n_face_loc, dtype=dtype)

  cnumbering.ngon_dconnectivity_from_gnum(*bounds, np.array(n_cell, dtype=dtype), face_pe, face_vtx)
  return face_vtx, face_pe
