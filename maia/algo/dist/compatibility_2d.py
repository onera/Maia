import numpy as np

import maia
from   maia.typing import *
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils
from .connectivity_utils import combine_face_edge_and_edge_vtx
from .ngon_tools         import PDM_dfacecell_to_dcellface

IS_2D_BASE = PT.pred.label_is('CGNSBase_t') & PT.pred.NodePredicate(lambda n : PT.get_np_value(n)[0] == 2)

def _convert_ngon2d_to_bar(zone):
  """
  Function to convert, in 2D, NGon node that wrongly describe edge_vtx 
  connectivity to BAR node
  """
  # > Get NGon node
  ngon_n = PT.Zone.NGonNode(zone)
  # > Test if only edges in ngon_n
  eso = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
  nb_nodes_per_faces = np.diff(eso)
  if nb_nodes_per_faces.size > 0:
    assert (nb_nodes_per_faces == 2).all()
  
  # > Del ESO
  PT.rm_node_from_path(ngon_n, 'ElementStartOffset')

  # > Change name and value (22 => 3)
  PT.update_node(ngon_n, name='EdgeElements', value=[3,0])

def _convert_nface2d_to_ngon(zone, comm):
  """
  Convert, in 2D, NFace node that wrongly describe face_edge 
  connectivity to NGon node describing face_vtx connectivity
  """
  # > Get Bar node information
  bar_n   = MT.Zone.EdgeNode(zone)
  nface_n = PT.Zone.NFaceNode(zone)
  
  # > Get vertex ids of faces from bar ids    
  face_bar_n = PT.get_child_from_name(nface_n, 'ElementConnectivity')
  face_vtx = combine_face_edge_and_edge_vtx(PT.get_child_from_name(nface_n, 'ElementStartOffset')[1],
                                            face_bar_n[1],
                                            MT.Element.distribution(bar_n),
                                            PT.get_child_from_name(bar_n, 'ElementConnectivity')[1],
                                            comm)
  PT.set_value(face_bar_n, face_vtx)
  
  # > Change name and value (23 => 22)
  PT.update_node(nface_n, name='NGonElements', value=[22,0])


def _bar_pe_to_nface2d(zone, comm):
  """
  Convert, in 2D, BAR node with ParentElements node to a NFace node
  that wrongly describe face_edge connectivity
  """
  PT.rm_child(zone, PT.Zone.NGonNode(zone))

  # > Get Bar node information
  bar_n = MT.Zone.EdgeNode(zone)
  PT.set_value(bar_n, [22, 0])
  
  edge_distrib = MT.Element.distribution(bar_n)
  face_distrib = MT.Zone.cell_distribution(zone)
  
  edge_face = maia.algo.indexing.get_pe_local(bar_n).reshape(-1,order='C')
  edge_face_idx = 2*np.arange(edge_distrib[0], edge_distrib[1]+1, dtype=np.int32)
  PT.new_DataArray('ElementStartOffset', value=edge_face_idx, parent=bar_n)
  
  full_edge_distrib = par_utils.partial_to_full_distribution(edge_distrib, comm)
  full_face_distrib = par_utils.partial_to_full_distribution(face_distrib, comm)
  face_edge = PDM_dfacecell_to_dcellface(comm,
                                         full_edge_distrib,
                                         full_face_distrib,
                                         edge_face)

  face_edge_distri = par_utils.dn_to_distribution(face_edge.dsize, comm)
  eso = face_edge.displs + face_edge_distri[0]

  nface_er = np.array([1, face_distrib[-1]], face_edge.dtype) + PT.Element.Range(bar_n)[1]
  nface_n = PT.new_NFaceElements(erange=nface_er, ec=face_edge.values, eso=eso, parent=zone)

  MT.new_Distribution({'Element': face_distrib}, parent = nface_n)



def poly2d_convert_3dlike_to_std(dist_tree, comm):
  """
  Convert a "as for 3D" 2D polyedric tree (NFACE = face_edge, NGON = edge_vtx)
  to the cgns compliant vision (NGON = face_vtx, BAR = edge_vtx).

  3like vision is supposed have correct edge orientations.

  Args:
    dist_tree (CGNSDistTree): Distributed tree
    comm      (MPIComm)     : MPI communicator
  """
  for zone in PT.get_children_from_predicates(dist_tree, [IS_2D_BASE, 'Zone_t']):
    _convert_ngon2d_to_bar(zone)
    _convert_nface2d_to_ngon(zone, comm)
    for subset in PT.iter_all_subsets(zone, 'FaceCenter'):
      PT.update_child(subset, 'GridLocation', 'GridLocation_t', 'EdgeCenter')


def poly2d_convert_std_to_3dlike(dist_tree, comm):
  """
  Convert a CGNS compliant 2D polyedric tree (NGON = face_vtx, BAR = edge_vtx)
  to the "as for 3D" vision (NFACE = face_edge, NGON = edge_vtx) used by legacy tools
  and some solvers

  Args:
    dist_tree (CGNSDistTree): Distributed tree
    comm      (MPIComm)     : MPI communicator
  """
  for zone in PT.get_children_from_predicates(dist_tree, [IS_2D_BASE, 'Zone_t']):
    _bar_pe_to_nface2d(zone, comm)
    for subset in PT.iter_all_subsets(zone, 'EdgeCenter'):
      PT.update_child(subset, 'GridLocation', 'GridLocation_t', 'FaceCenter')
