from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.typing import *

import maia

from maia.algo.dist.geometry import _compute_elements_center, _compute_elements_normal
from maia.transfer import protocols as EP

IS_PHYDIM_2 = PT.pred.NodePredicate(lambda z: PT.Zone.PhysicalDimension(z) == 2)

def _remove_z(array:NDArray) -> NDArray: 
  assert array.size % 3 == 0
  remove_mask = np.tile([True, True, False], array.size // 3)
  return array[remove_mask]

def iter_matching_zones(t: CGNSTree, cond: Callable[[CGNSTree], bool]) -> Iterator[CGNSTree]:
  for z in PT.iter_all_Zone_t(t):
    if cond(z):
      yield z

def enforce_boundary_pe_left(tree:CGNSDistTree, comm:MPIComm) -> None:
  """
  Force the boundary ngon to have a non zero left parent cell.
  After this operation, the normal of the face may be badly oriented:
  use fix_normal_orientation to correct that.

  This function only update polyedric zones
  """

  maia.algo.nface_to_pe(tree, comm)
  maia.algo.ngon_to_edge_pe(tree, comm)

  is_poly = PT.pred.IS_POLY2D_ZONE | PT.pred.IS_POLY3D_ZONE

  for zone in iter_matching_zones(tree, is_poly):

    get_fn = PT.Zone.NGonNode if PT.Zone.CellDimension(zone) == 3 else MT.Zone.EdgeNode
    bnd_elt_node = get_fn(zone)
    
    pe_n = PT.find_child_from_name(bnd_elt_node, 'ParentElements')
    pe = PT.get_np_value(pe_n)

    need_swap = pe[:,0] == 0
    
    # Early return if all bnd faces have already left parent
    if not comm.allreduce(need_swap.any(), MPI.LOR):
      return

    # Swap PE
    pe[need_swap, 0] = pe[need_swap, 1]
    pe[need_swap, 1] = 0

    # Change sign in NFace (only for 3d zones; for 2d zones, the orientation of edges
    # does not impact the NGON node so we don't do anything)
    if PT.Zone.has_nface_elements(zone):
      face_distri   = MT.Element.distribution(bnd_elt_node)
      nface_node = PT.Zone.NFaceNode(zone)
      cell_face = PT.get_np_value(PT.find_child_from_name(nface_node, 'ElementConnectivity'))
      GI = EP.GlobalIndexer(face_distri, abs(cell_face)-PT.Element.Range(bnd_elt_node)[0], comm)
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

  This function only update polyedric zones
  """

  # Ensure NFace & PE are both present (these fonctions select relevant zones only)
  maia.algo.pe_to_nface(tree, comm)
  maia.algo.nface_to_pe(tree, comm)
  maia.algo.edge_pe_to_ngon(tree, comm)
  maia.algo.ngon_to_edge_pe(tree, comm)

  is_poly = PT.pred.IS_POLY3D_ZONE | (PT.pred.IS_POLY2D_ZONE & IS_PHYDIM_2)
  for zone in iter_matching_zones(tree, is_poly):
    cell_dim = PT.Zone.CellDimension(zone)
    phy_dim = PT.Zone.PhysicalDimension(zone)
    
    # *NB* In all this function we use the words:
    #   ngon = face and nface = cell for 3D meshes
    #   ngon = edge and nface = face for 2D meshes

    ngon_node  = PT.Zone.NGonNode(zone)  if cell_dim == 3 else MT.Zone.EdgeNode(zone)
    nface_node = PT.Zone.NFaceNode(zone) if cell_dim == 3 else PT.Zone.NGonNode(zone)
    pe_n = PT.find_child_from_name(ngon_node, 'ParentElements')
    pe   = PT.get_np_value(pe_n)

    # Compute face normals and cell centers
    cell_center = _compute_elements_center(zone, 'CellCenter', comm)
    face_normal = _compute_elements_normal(zone, comm)
    assert cell_center is not None
    assert face_normal is not None

    # If working on 2D cases, extract xy component from cell_center
    # (face center is already xy only)
    if phy_dim == 2:
      cell_center = _remove_z(cell_center)

    # Detect boundary faces from PE
    boundary_flag = (pe[:,1] != 0)

    # To mark the faces for which normal is badly oriented
    flip_mask = np.empty(pe.shape[0], dtype=bool)

    # > Internal faces

    # For each internal face (resp. edge), get the center of the left and right cell (resp. face)
    cell_distri   = MT.Zone.cell_distribution(zone)
    parents_center = EP.GlobalIndexer(cell_distri,
                                      np.ravel(pe[boundary_flag]) - PT.Element.Range(nface_node)[0],
                                      comm).Take(cell_center, count=phy_dim)

    # Compute Right - Left vector for each face
    parents_center.shape = (-1, 2*phy_dim)
    centers_vector = parents_center[:, phy_dim:] - parents_center[:, :phy_dim]

    # Now compute scalar product
    internal_face_normal = face_normal.reshape((-1,phy_dim))[boundary_flag]
    scalar_prod = (centers_vector * internal_face_normal).sum(axis=1)

    flip_mask[boundary_flag] = (scalar_prod < 0)

    # > External faces
    np.invert(boundary_flag, out=boundary_flag)

    # Compute face center, only for external faces
    face_distri = MT.Element.distribution(ngon_node)
    external_face_pl = np.arange(face_distri[0], face_distri[1])[boundary_flag] + PT.Element.Range(ngon_node)[0]
    external_face_pl = external_face_pl.reshape((1,-1), order='F')
    external_faces_center = _compute_elements_center(zone, phy_dim-1, comm, external_face_pl)

    # Again
    if phy_dim == 2:
      external_faces_center = _remove_z(external_faces_center)

    # Get the center of left cell only
    parent_center = EP.GlobalIndexer(cell_distri,
                                     pe[boundary_flag, 0] - PT.Element.Range(nface_node)[0],
                                     comm).Take(cell_center, count=phy_dim)
    # Compute Face - Left cell vector for each face
    centers_vector = external_faces_center - parent_center
    centers_vector.shape = (-1, phy_dim)

    # Now compute scalar product
    external_face_normal = face_normal.reshape((-1,phy_dim))[boundary_flag]
    scalar_prod = (centers_vector * external_face_normal).sum(axis=1)
    
    flip_mask[boundary_flag] = (scalar_prod < 0)

    # Finally, flip the connectivity of face badly oriented (PE & NF are not modified)
    face_vtx = MT.Element.connectivity(ngon_node)
    face_vtx._inner_flip(flip_mask)

