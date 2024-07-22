import numpy as np

import Pypdm.Pypdm as PDM

from maia.transfer  import protocols  as EP
from maia.utils     import np_utils



def combine_face_edge_and_edge_vtx(face_edge_idx, face_edge, edge_distrib, edge_vtx, comm):
  """
  Compute face_vtx connectivity from face_edge and edge_vtx connectivities in
  a distributed context

  Args:
    face_edge_idx (array)  : Face to edge connectivity index
    face_edge     (array)  : Face to edge connectivity
    edge_distrib  (array)  : Edge distribution
    edge_vtx      (array)  : Edge to vertex connectivity
    comm          (MPIComm): MPI communicator
  """
  face_edge_idx = np_utils.safe_int_cast(face_edge_idx - face_edge_idx[0], np.int32)
  
  dist_data = {'connectivity' : edge_vtx}
  dist_stride = np.ones(edge_distrib[1]-edge_distrib[0], dtype=np.int32) * 2
  part_stride, part_data = EP.block_to_part_strided(dist_stride, dist_data, edge_distrib, [face_edge], comm)
  global_edge_vtx = part_data["connectivity"][0]
  
  # Convert in local numbering to be allowed to use part algo in dist context
  # We basically redefine all edges (internal edges are defined twice) instead of creating 
  # a proper subnumbering, it is ok because then we extract the resulting vertices
  uniq, idx, inv = np.unique(global_edge_vtx, return_index=True, return_inverse=True)
  local_edge_vtx = np_utils.safe_int_cast(inv+1, np.int32)
  
  local_face_edge = np.sign(face_edge, dtype=np.int32) * np.arange(1, face_edge.size+1, dtype=np.int32)
  
  local_face_vtx = PDM.compute_face_vtx_from_face_and_edge(face_edge_idx,
                                                           local_face_edge,
                                                           local_edge_vtx)
  
  # Return in the global numbering
  global_face_vtx = global_edge_vtx[idx][local_face_vtx-1]
  
  return global_face_vtx
