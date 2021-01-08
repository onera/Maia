"""
These numbering functions map cells, faces or vertex of a structured mesh,
identified by their three indexes (i,j,k), to an absolute 1d identifier.
Some conventions shared by all the functions:
- (i,j,k) and resultting indexes start at 1
- n_cell / n_vtx is the number of cells and vertices in the structured block
- The face numerotation starts with all faces with normal in direction
  i, then continues with all faces with normal in direction j and ends
  with all faces with normal in direction k

"""
###############################################################################
def ijk_to_index(i,j,k,n_elmt):
  """
  Convert (i,j,k) indices from structured grid to unstructured index
  This fonction allows (i,j,k) to describe a vertex or a cell, depending if
  n_elmt is the number of nodes or cells of the structured block
  """
  return(i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1])
###############################################################################

###############################################################################
def ijk_to_faceiIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction i
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1)
  """
  return(i + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_cell[1])
###############################################################################

###############################################################################
def ijk_to_facejIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction j
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k)
  """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  return(i + (j-1)*n_cell[0] + (k-1)*n_vtx[1]*n_cell[0] + nbFacesi)
###############################################################################

###############################################################################
def ijk_to_facekIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction k
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k)
  """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  nbFacesj = n_vtx[1]*n_cell[0]*n_cell[2]
  return(i + (j-1)*n_cell[0] + (k-1)*n_cell[0]*n_cell[1] + nbFacesi + nbFacesj)
###############################################################################

###############################################################################
def ijk_to_faceIndices(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of faces with
  normal in direction i, j and k
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1) for fi
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k) for fj
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k) for fk
  """
  fi = ijk_to_faceiIndex(i,j,k,n_cell,n_vtx)
  fj = ijk_to_facejIndex(i,j,k,n_cell,n_vtx)
  fk = ijk_to_facekIndex(i,j,k,n_cell,n_vtx)
  return(fi,fj,fk)
###############################################################################

###############################################################################
def compute_fi_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction i and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1) if i is not min
  (i,j,k), (i,j,k+1), (i,j+1,k+1) and (i,j+1,k) if i is min
  """
  # Nodes of the face
  n1 = (i,j  ,k  )
  n2 = (i,j+1,k  )
  n3 = (i,j+1,k+1)
  n4 = (i,j  ,k+1)
  # Neighbour cells of the face
  left  = (i-1,j,k)
  right = (i  ,j,k)
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################

###############################################################################
def compute_fj_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction j and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k) if j is not min
  (i,j,k), (i+1,j,k), (i+1,j,k+1) and (i,j,k+1) if j is min
  """
  # Nodes of the face
  n1 = (i  ,j,k  )
  n2 = (i  ,j,k+1)
  n3 = (i+1,j,k+1)
  n4 = (i+1,j,k  )
  # Neighbour cells of the face
  left  = (i,j-1,k)
  right = (i,j  ,k)
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################

###############################################################################
def compute_fk_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction k and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k) if k is not min
  (i,j,k), (i,j+1,k), (i+1,j+1,k) and (i+1,j,k) if k is min
  """
  # Nodes of the face
  n1 = (i  ,j  ,k)
  n2 = (i+1,j  ,k)
  n3 = (i+1,j+1,k)
  n4 = (i  ,j+1,k)
  # Neighbour cells of the face
  left  = (i,j,k-1)
  right = (i,j,k  )
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################
