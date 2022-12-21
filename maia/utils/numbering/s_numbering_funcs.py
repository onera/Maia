import numpy as np

from cmaia.utils import numbering as cnumbering
"""
These numbering functions map cells, faces or vertex of a structured mesh,
identified by their three indices (i,j,k), to an absolute 1d identifier.
Some conventions shared by all the functions:
- (i,j,k) and resulting indices start at 1
- n_cell / n_vtx is the number of cells and vertices in the structured block
- The face numbering starts with all faces with normal in direction
  i, then continues with all faces with normal in direction j and ends
  with all faces with normal in direction k
- Functions should be call directly on numpy arrays for optimal performance
"""
###############################################################################

def ijk_to_index_from_loc(i,j,k, loc, n_vtx):
  n_cell = n_vtx - 1
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
  raise ValueError("Unsupported location")

def index_to_ijk_from_loc(idx, loc, n_vtx):
  n_cell = n_vtx - 1
  if loc == 'Vertex':
    return index_to_ijk(idx, n_vtx)
  elif loc == 'CellCenter':
    return index_to_ijk(idx, n_cell)
  elif loc == 'IFaceCenter':
    return faceiIndex_to_ijk(idx, n_cell, n_vtx)
  elif loc == 'JFaceCenter':
    return ijk_to_facejIndex(i,j,k,n_cell,n_vtx)
    return facejIndex_to_ijk(idx, n_cell, n_vtx)
  elif loc == 'KFaceCenter':
    return facekIndex_to_ijk(idx, n_cell, n_vtx)
  raise ValueError("Unsupported location")


def ijk_to_index(i,j,k,n_elmt):
  """
  Convert (i,j,k) indices from structured grid to unstructured index
  This fonction allows (i,j,k) to describe a vertex or a cell, depending if
  n_elmt is the number of nodes or cells of the structured block
  """
  return(i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1])

def index_to_ijk(idx, n_elmt):
  """
  Convert an index to a i,j,k triplet. This fonction allows the index to describe a
  vertex or a cell, depending if n_elmt is the number of nodes or cells of the
  structured block
  """
  k = ((idx - 1) // (n_elmt[0]*n_elmt[1])) + 1
  j = (idx - (k-1)*(n_elmt[0]*n_elmt[1]) - 1) // n_elmt[0] + 1
  i = idx - (j-1)*n_elmt[0] - (k-1)*(n_elmt[0]*n_elmt[1])
  return i,j,k

def ijk_to_faceiIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction i
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1)
  """
  return(i + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_cell[1])

def faceiIndex_to_ijk(idx, n_cell, n_vtx):
  """ Convert an index of I-normal face to i,j,k triplet """
  k = ((idx - 1) // (n_vtx[0]*n_cell[1])) + 1
  j = (idx - (k-1)*(n_vtx[0]*n_cell[1]) - 1) // n_vtx[0] + 1
  i = idx - (j-1)*n_vtx[0] - (k-1)*(n_vtx[0]*n_cell[1])
  return i,j,k

def ijk_to_facejIndex(i,j,k,n_cell,n_vtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction j
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k)
  """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  return(i + (j-1)*n_cell[0] + (k-1)*n_vtx[1]*n_cell[0] + nbFacesi)

def facejIndex_to_ijk(idx, n_cell, n_vtx):
  """ Convert an index of J-normal face to i,j,k triplet """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  k = ((idx - 1 - nbFacesi) // (n_vtx[1]*n_cell[0])) + 1
  j = (idx - (k-1)*(n_vtx[1]*n_cell[0]) - 1 - nbFacesi) // n_cell[0] + 1
  i = idx - (j-1)*n_cell[0] - (k-1)*(n_vtx[1]*n_cell[0]) - nbFacesi
  return i,j,k

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

def facekIndex_to_ijk(idx, n_cell, n_vtx):
  """ Convert an index of K-normal face to i,j,k triplet """
  nbFacesi = n_vtx[0]*n_cell[1]*n_cell[2]
  nbFacesj = n_vtx[1]*n_cell[0]*n_cell[2]
  k = ((idx - 1 - nbFacesi - nbFacesj) // (n_cell[0]*n_cell[1])) + 1
  j = (idx - (k-1)*(n_cell[0]*n_cell[1]) - 1 - nbFacesi - nbFacesj) // n_cell[0] + 1
  i = idx - (j-1)*n_cell[0] - (k-1)*(n_cell[0]*n_cell[1]) - nbFacesi - nbFacesj
  return i,j,k

###############################################################################

###############################################################################
def PE_idx_from_i_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of left and right parents of a i-normal face directly from
  its index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[i_face_id,:] = [left_parent_id, right_parent_id]
  """
  return cnumbering.facepe_from_i_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def facevtx_from_i_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of the 4 nodes belonging to a i-normal face directly from
  its index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*i_face_id:4*(i_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  return cnumbering.facevtx_from_i_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def PE_idx_from_j_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of left and right parents of a j-normal face directly from
  its index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[j_face_id,:] = [left_parent_id, right_parent_id]
  """
  return cnumbering.facepe_from_j_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def facevtx_from_j_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of the 4 nodes belonging to a j-normal face directly from
  its index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*j_face_id:4*(j_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  return cnumbering.facevtx_from_j_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def PE_idx_from_k_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of left and right parents of a k-normal face directly from
  its index. Returns a (#idx, 2) shaped array PE storing for each face
  PE[k_face_id,:] = [left_parent_id, right_parent_id]
  """
  return cnumbering.facepe_from_k_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def facevtx_from_k_face_idx(idx, n_cell, n_vtx):
  """
  Compute the index of the 4 nodes belonging to a k-normal face directly from
  its index. Returns a flattened array nodes of size 4*#idx storing for each face
  nodes[4*k_face_id:4*(k_face_id+1)] = [n1,n2,n3,n4]. Nodes are ordered such that
  face normal goes from leftcell to rightcell.
  """
  return cnumbering.facevtx_from_k_face(np.asarray(idx), np.array(n_cell, idx.dtype))

def ngon_dconnectivity_from_gnum(bounds, n_cell, dtype):
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
