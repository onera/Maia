import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo import indexing
from maia.utils import np_utils
from maia.utils import vstride as vs

import cmaia.part_algo as cpart_algo

def PDM_face_vtx_from_face_and_edge(face_edge_idx, face_edge, edge_vtx):
  # Cast are not necessary since partitionned meshes are supposed to be int32
  _face_edge_idx = np_utils.safe_int_cast(face_edge_idx, np.int32)
  _face_edge     = np_utils.safe_int_cast(face_edge, np.int32)
  _edge_vtx      = np_utils.safe_int_cast(edge_vtx, np.int32)
  _face_vtx      = PDM.compute_face_vtx_from_face_and_edge(_face_edge_idx, _face_edge, _edge_vtx)
  return np_utils.safe_int_cast(_face_vtx, dtype=face_edge.dtype)


def pe_to_nface(zone, remove_PE=False):
  """Create a NFace node from a NGon node with ParentElements.

  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Partitioned zone
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """
  ngon_node = PT.Zone.NGonNode(zone)
  er = PT.get_child_from_name(ngon_node, 'ElementRange')[1]
  pe = PT.get_child_from_name(ngon_node, 'ParentElements')[1]
  max_cell = np.max(pe)
  min_cell = np.min(pe[np.nonzero(pe)])

  local_pe = indexing.get_pe_local(ngon_node)

  nface_eso, nface_ec = cpart_algo.local_pe_to_local_cellface(local_pe) #Compute NFace connectivity

  #Put NFace/EC in global numbering (to refer to ngon global ids)
  first_ngon = er[0]
  if first_ngon != 1:
    nface_ec_sign = np.sign(nface_ec)
    nface_ec_sign*(np.abs(nface_ec) + first_ngon - 1)

  #Create NFace node
  _erange = np.array([min_cell, max_cell], dtype=np.int32)
  nface = PT.new_NFaceElements(erange=_erange, eso=nface_eso, ec=nface_ec, parent=zone)
  cell_gnum = MT.getGlobalNumbering(zone, 'Cell')
  if cell_gnum is not None:
    MT.newGlobalNumbering({'Element' : PT.get_value(cell_gnum)}, nface)

  if remove_PE:
    PT.rm_children_from_name(ngon_node, "ParentElements")


def nface_to_pe(zone, remove_NFace=False):
  """Create a ParentElements node in the NGon node from a NFace node.

  Input tree is modified inplace.

  Args:
    zone         (CGNSTree): Partitioned zone
    remove_NFace (bool, optional): If True, remove the NFace node.
      Defaults to False.
  """
  ngon_node  = PT.Zone.NGonNode(zone)
  nface_node = PT.Zone.NFaceNode(zone)

  cell_face_idx = PT.get_child_from_name(nface_node, "ElementStartOffset")[1]
  cell_face     = PT.get_child_from_name(nface_node, "ElementConnectivity")[1]

  # If NFace are before NGon, then face ids must be shifted
  if PT.Element.Range(ngon_node)[0] == 1:
    _cell_face = cell_face
  else:
    _cell_face_sign = np.sign(cell_face)
    _cell_face = np.abs(cell_face) - PT.Element.Size(nface_node)
    _cell_face = _cell_face * _cell_face_sign

  local_pe = cpart_algo.local_cellface_to_local_pe(cell_face_idx, _cell_face)
  np_utils.shift_nonzeros(local_pe, PT.Element.Range(nface_node)[0]-1) # Refer to NFace global ids

  PT.new_DataArray('ParentElements', local_pe, parent=ngon_node)
  if remove_NFace:
    PT.rm_child(zone, nface_node)


def ngon_to_edge_pe(zone, remove_NGon=False):
  """Create a ParentElements node in the EdgeElements node from a NGon node.

  Note that EdgeElement is supposed to exists and define all (including internal)
  edges. This function computes the link between these edges and the NGon node.

  Input tree is modified inplace.

  Args:
    zone         (CGNSTree): Partitioned zone
    remove_NGon (bool, optional): If True, remove the NGon node.
      Defaults to False.
  """
  # EDGE Data
  edge_node  = MT.Zone.EdgeNode(zone)
  edge_vtx = PT.get_child_from_name(edge_node, 'ElementConnectivity')[1]
  key_from_edge = edge_vtx[0::2] + edge_vtx[1::2] # hash vtx-vtx of BAR_2 by taking the sum of both

  # NGON Data
  ngon_node = PT.Zone.NGonNode(zone)
  face_vtx  = MT.Element.connectivity(ngon_node)

  first_vtx  = face_vtx
  second_vtx = vs.roll(first_vtx, -1, vs.INNER_AXIS)

  key_from_face = first_vtx.values + second_vtx.values # hash by vtx-vtx sum as above
  start_face_id = PT.Element.Range(ngon_node)[0]
  edge_parent_id = np_utils.repeated_arange(face_vtx.counts, start_face_id, dtype=face_vtx.dtype)


  # Now do the search using key
  # First : unify data from face into a block-like vision ie
  #  sort related data and transform key in unique + counts
  sort_idx = np.argsort(key_from_face)
  key_from_face  = key_from_face[sort_idx]
  edge_parent_id = edge_parent_id[sort_idx]
  face_first_vtx = first_vtx.values[sort_idx]

  key_from_face_unique, key_from_face_counts = np_utils.unique_sorted(key_from_face, return_counts=True)
  key_from_face_idx = np_utils.sizes_to_indices(key_from_face_counts)
  

  # Second : get data from block-like vision, for each edge

  # Index of edge key in face key sorted array
  select_idx = np.searchsorted(key_from_face_unique, key_from_edge)

  counts_for_edge = key_from_face_counts[select_idx] # get the number collisions by edge (count==1 <=> no collision)
  parent_id_for_edge = vs.take(vs.from_displs(key_from_face_idx, edge_parent_id), select_idx).values
  first_vtx_for_edge = vs.take(vs.from_displs(key_from_face_idx, face_first_vtx), select_idx).values


  # Third: post treat (solving conflicts) for fill edge_face
  n_edge = edge_vtx.size // 2
  edge_face = np.zeros((n_edge, 2), order='F', dtype=edge_vtx.dtype)

  iedge_extended = np_utils.repeated_arange(counts_for_edge)

  # Test if vertex of each edges is equal to recv face_first_vtx, because we can
  # have the same key for several edges pairs
  first_vtx_match  = edge_vtx[2*iedge_extended  ] == first_vtx_for_edge
  second_vtx_match = edge_vtx[2*iedge_extended+1] == first_vtx_for_edge
  # Fill edge_face with matching edge_parent_id
  edge_face[iedge_extended[first_vtx_match],  0] = parent_id_for_edge[first_vtx_match]
  edge_face[iedge_extended[second_vtx_match], 1] = parent_id_for_edge[second_vtx_match]


  PT.new_DataArray('ParentElements', edge_face, parent=edge_node)
  if remove_NGon:
    PT.rm_child(zone, ngon_node)


def edge_pe_to_ngon(zone, remove_PE=False):
  """Create a NGon node from a Edge node with ParentElements.

  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Partitioned zone
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """

  edge_node = MT.Zone.EdgeNode(zone)
  pe = PT.get_child_from_name(edge_node, 'ParentElements')[1]
  max_face = np.max(pe)
  min_face = np.min(pe[np.nonzero(pe)])
  local_pe = indexing.get_pe_local(edge_node)
  edge_vtx = PT.get_child_from_name(edge_node, 'ElementConnectivity')[1]

  ngon_eso, face_edge = cpart_algo.local_pe_to_local_cellface(local_pe)
  ngon_ec = PDM_face_vtx_from_face_and_edge(ngon_eso, face_edge, edge_vtx)
  _erange = np.array([min_face, max_face], dtype=np.int32)
  ngon = PT.new_NGonElements(erange=_erange, eso=ngon_eso, ec=ngon_ec, parent=zone)
  face_gnum = MT.getGlobalNumbering(zone, 'Cell') # cell = face
  if face_gnum is not None:
    MT.newGlobalNumbering({'Element' : PT.get_value(face_gnum)}, ngon)

  if remove_PE:
    PT.rm_children_from_name(edge_node, "ParentElements")
