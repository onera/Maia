from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.typing import *

import maia

from maia.algo.dist.geometry import _compute_elements_center, _compute_elements_normal
from maia.transfer import protocols as EP

from maia.utils import par_utils

def enforce_boundary_pe_left(tree:CGNSDistTree, comm:MPIComm) -> None:
  """
  Force the boundary ngon to have a non zero left parent cell.
  In such case, connetivities (FaceVtx & NFace, if existing) are reversed to preserve face
  orientation.
  This function only accepts 3D NGON trees.
  """
  for zone in PT.iter_all_Zone_t(tree):
    assert PT.Zone.CellDimension(zone) == 3, "Only 3D meshes are supported"
    
    ngon_node = PT.Zone.NGonNode(zone)
    if PT.get_child_from_name(ngon_node, 'ParentElements') is None:
      maia.algo.nface_to_pe(zone, comm)
    pe_n = PT.find_child_from_name(ngon_node, 'ParentElements')
    pe = PT.get_np_value(pe_n)

    need_swap = pe[:,0] == 0
    
    # Early return if all bnd faces have already left parent
    if not comm.allreduce(need_swap.any(), MPI.LOR):
      return

    # Swap PE
    pe[need_swap, 0] = pe[need_swap, 1]
    pe[need_swap, 1] = 0

    # Swap NG connectivity
    face_vtx = MT.Element.connectivity(ngon_node)
    face_vtx._inner_flip(need_swap)

    # Change sign in NFace
    if PT.Zone.has_nface_elements(zone):
      face_distri   = MT.distribution_value(ngon_node, 'Element')
      face_distri_f = par_utils.partial_to_full_distribution(face_distri, comm) 
      nface_node = PT.Zone.NFaceNode(zone)
      cell_face = PT.get_np_value(PT.find_child_from_name(nface_node, 'ElementConnectivity'))
      GI = EP.GlobalIndexer(face_distri_f, abs(cell_face)-PT.Element.Range(ngon_node)[0], comm)
      need_swap_loc = GI.Take(need_swap)
      np.multiply(cell_face, -1, out=cell_face, where=need_swap_loc)

def fix_normal_orientation(tree:CGNSDistTree, comm:MPIComm) -> None:
  """
  Invert the normal (by swapping face_vtx connectivity) of faces that does
  not respect the following convention:
    - left parent to right parent orientation for internal faces
    - outward orientation for external faces

  Note that:
    - This function assumes that external faces have a left parent. Use enforce_boundary_pe_left if 
      it is not the case
    - This function relies on a geometric test (dot product)

  This function applies to polyedric 3D meshes only
  """

  # Ajouter check ngon 3D
  for zone in PT.iter_all_Zone_t(tree):
    assert PT.Zone.CellDimension(zone) == 3, "Only 3D meshes are supported"

    # Ensure NFace & PE are both present
    maia.algo.pe_to_nface(zone, comm)
    maia.algo.nface_to_pe(zone, comm)

    ngon_node  = PT.Zone.NGonNode(zone)
    nface_node = PT.Zone.NFaceNode(zone)
    pe_n = PT.find_child_from_name(ngon_node, 'ParentElements')
    pe   = PT.get_np_value(pe_n)

    PT.new_FlowSolution
    # Compute face normals and cell centers
    cell_center = _compute_elements_center(zone, 3, comm)
    face_normal = _compute_elements_normal(zone, comm)
    # CellDim    phydim=3     phydim=2    phydim=1
    # Volumic       Face           KO           KO             
    # Surfacic      Face        Edge            KO
    # Lineic                         
    #
    assert cell_center is not None
    assert face_normal is not None

    # Detect boundary faces from PE
    boundary_flag = (pe[:,1] != 0)

    # To mark the faces for which normal is badly oriented
    flip_mask = np.empty(pe.shape[0], dtype=bool)

    # > Internal faces

    # For each internal face, get the center of the left and right cell
    cell_distri   = MT.distribution_value(zone, 'Cell')
    cell_distri_f = par_utils.partial_to_full_distribution(cell_distri, comm)
    parents_center = EP.GlobalIndexer(cell_distri_f,
                                      np.ravel(pe[boundary_flag]) - PT.Element.Range(nface_node)[0],
                                      comm).Take(cell_center, count=3)

    # Compute Right - Left vector for each face
    parents_center.shape = (-1, 6)
    centers_vector = parents_center[:, 3:] - parents_center[:, :3]

    # Now compute scalar product
    internal_face_normal = face_normal.reshape((-1,3))[boundary_flag]
    scalar_prod = (centers_vector * internal_face_normal).sum(axis=1)

    flip_mask[boundary_flag] = (scalar_prod < 0)

    # > External faces
    np.invert(boundary_flag, out=boundary_flag)

    # Compute face center, only for external faces
    face_distri = MT.distribution_value(ngon_node, 'Element')
    external_face_pl = np.arange(face_distri[0], face_distri[1])[boundary_flag] + PT.Element.Range(ngon_node)[0]
    external_face_pl = external_face_pl.reshape((1,-1), order='F')
    external_faces_center = _compute_elements_center(zone, 2, comm, external_face_pl)

    # Get the center of left cell only
    parent_center = EP.GlobalIndexer(cell_distri_f,
                                     pe[boundary_flag, 0] - PT.Element.Range(nface_node)[0],
                                     comm).Take(cell_center, count=3)
    # Compute Face - Left cell vector for each face
    centers_vector = external_faces_center - parent_center
    centers_vector.shape = (-1, 3)

    # Now compute scalar product
    external_face_normal = face_normal.reshape((-1,3))[boundary_flag]
    scalar_prod = (centers_vector * external_face_normal).sum(axis=1)
    
    flip_mask[boundary_flag] = (scalar_prod < 0)

    # Finally, flip the connectivity of face badly oriented (PE & NF are not modified)
    face_vtx = MT.Element.connectivity(ngon_node)
    face_vtx._inner_flip(flip_mask)

