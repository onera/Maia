import numpy as np
"""
These numbering functions map cells, faces or vertex of a structured mesh,
identified by their three indexes (i,j,k), to an absolute 1d identifier.
Some conventions shared by all the functions:
- (i,j,k) and resultting indexes start at 1
- n_cell / n_vtx is the number of cells and vertices in the structured block
- The face numerotation starts with all faces with normal in direction
  i, then continues with all faces with normal in direction j and ends
  with all faces with normal in direction k
- Functions should be call directly on numpy arrays for optimal performance
"""
###############################################################################
def ijk_to_index(i,j,k,n_elmt):
  """
  Convert (i,j,k) indices from structured grid to unstructured index
  This fonction allows (i,j,k) to describe a vertex or a cell, depending if
  n_elmt is the number of nodes or cells of the structured block
  """
  return(i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1])

def ijk_to_faceiIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction i
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1)
  """
  return(i + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_cell[1])

def ijk_to_facejIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction j
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k)
  """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  return(i + (j-1)*n_cell[0] + (k-1)*n_vtx[1]*n_cell[0] + nbFacesi)

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
def compute_fi_PE_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of left and right parents of a i-normal face directly from
  its U index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[i_face_id,:] = [left_parent_id, right_parent_id]
  """
  np_idx = np.asarray(idx)
  is_min_bnd  = (np_idx % n_vtx[0]) == 1
  is_max_bnd  = (np_idx % n_vtx[0]) == 0
  is_internal = ~(is_min_bnd | is_max_bnd)
  line_number = (np_idx-1) // n_vtx[0]

  #Internal faces : left, right = idx-line_number-1, idx-line_number
  #Min faces      : left        = idx-line_number
  #Max faces      : left        = idx-line_number-1
  PE = np.empty((np_idx.shape[0], 2), dtype=np_idx.dtype)
  PE[:,0] = np_idx - line_number - 1 + is_min_bnd
  PE[:,1] = (np_idx - line_number) * is_internal
  return PE

def compute_fi_facevtx_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of the 4 nodes belonging to a i-normal face directly from
  its U index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*i_face_id:4*(i_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  np_idx = np.asarray(idx)
  is_min_bnd  = (np_idx % n_vtx[0]) == 1
  plan_number = (np_idx-1) // (n_vtx[0]*n_cell[1])

  n1 = np_idx + plan_number*n_vtx[0]
  nodes = np.empty((np_idx.shape[0],4), dtype=np_idx.dtype)
  nodes[:,0] = n1
  nodes[:,1] = n1 + n_vtx[0]
  nodes[:,2] = n1 + n_vtx[0] + n_vtx[0]*n_vtx[1]
  nodes[:,3] = n1 + n_vtx[0]*n_vtx[1]
  #Swap 2,4 for min faces
  nodes[is_min_bnd, 1], nodes[is_min_bnd,3] = nodes[is_min_bnd, 3], nodes[is_min_bnd,1]
  return nodes.flatten()

def compute_fj_PE_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of left and right parents of a j-normal face directly from
  its U index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[j_face_id,:] = [left_parent_id, right_parent_id]
  """
  np_idx = np.asarray(idx)
  nb_face_ij  = n_vtx[1]*n_cell[0]
  plan_number = (np_idx-1) // nb_face_ij
  is_min_bnd  = (np_idx - plan_number*nb_face_ij) < n_vtx[0]
  is_max_bnd  = (np_idx - plan_number*nb_face_ij) > nb_face_ij-n_vtx[0]+1
  is_internal = ~(is_min_bnd | is_max_bnd)

  #Internal faces : left, right = idx - n_cell[0]*plan_number-n_cell[0], idx - n_cell[0]*plan_number
  #Min faces      : left        = idx - n_cell[0]*plan_number
  #Max faces      : left        = idx - n_cell[0]*plan_number-n_cell[0]
  PE = np.empty((np_idx.shape[0], 2), dtype=np_idx.dtype)
  PE[:,0] = np_idx - n_cell[0]*plan_number - n_cell[0]*(1-is_min_bnd)
  PE[:,1] = (np_idx - n_cell[0]*plan_number)*is_internal
  return PE

def compute_fj_facevtx_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of the 4 nodes belonging to a j-normal face directly from
  its U index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*j_face_id:4*(j_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  np_idx = np.asarray(idx)
  nb_face_ij  = n_vtx[1]*n_cell[0]
  line_number = (np_idx-1) // n_cell[0]
  plan_number = (np_idx-1) // nb_face_ij
  is_min_bnd  = (np_idx - plan_number*nb_face_ij) < n_vtx[0]

  n1 = np_idx + line_number
  nodes = np.empty((np_idx.shape[0],4), dtype=np_idx.dtype)
  nodes[:,0] = n1
  nodes[:,1] = n1 + n_vtx[0]*n_vtx[1]
  nodes[:,2] = n1 + n_vtx[0]*n_vtx[1] + 1
  nodes[:,3] = n1 + 1
  #Swap 2,4 for min faces
  nodes[is_min_bnd, 1], nodes[is_min_bnd,3] = nodes[is_min_bnd, 3], nodes[is_min_bnd,1]
  return nodes.flatten()

def compute_fk_PE_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of left and right parents of a k-normal face directly from
  its U index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[k_face_id,:] = [left_parent_id, right_parent_id]
  """
  np_idx = np.asarray(idx)
  nb_face_ij  = n_cell[0]*n_cell[1]
  is_min_bnd  = np_idx <= nb_face_ij
  is_max_bnd  = np_idx  > nb_face_ij*(n_vtx[2]-1)
  is_internal = ~(is_min_bnd | is_max_bnd)

  #Internal faces : left, right = idx - nb_face_ij, idx
  #Min faces      : left        = idx
  #Max faces      : left        = idx - nb_face_ij
  PE = np.empty((np_idx.shape[0], 2), dtype=np_idx.dtype)
  PE[:,0] = np_idx - nb_face_ij*(1-is_min_bnd)
  PE[:,1] = np_idx*is_internal
  return PE

def compute_fk_facevtx_from_idx(idx, n_cell, n_vtx):
  """
  Compute the U index of the 4 nodes belonging to a k-normal face directly from
  its U index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*k_face_id:4*(k_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  np_idx = np.asarray(idx)
  nb_face_ij  = n_cell[0]*n_cell[1]
  is_min_bnd  = np_idx <= nb_face_ij
  line_number = (np_idx - 1 ) // n_cell[0]
  plan_number = (np_idx - 1 ) // nb_face_ij

  n1 = np_idx + line_number + n_vtx[0]*plan_number
  nodes = np.empty((np_idx.shape[0],4), dtype=np_idx.dtype)
  nodes[:,0] = n1
  nodes[:,1] = n1 + 1
  nodes[:,2] = n1 + n_vtx[0] + 1
  nodes[:,3] = n1 + n_vtx[0]
  #Swap 2,4 for min faces
  nodes[is_min_bnd, 1], nodes[is_min_bnd,3] = nodes[is_min_bnd, 3], nodes[is_min_bnd,1]
  return nodes.flatten()

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
