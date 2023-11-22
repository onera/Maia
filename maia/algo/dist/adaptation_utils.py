import maia
import maia.pytree as PT
from   maia.utils  import np_utils, py_utils

import cmaia.dist_algo as cdist_algo

import numpy as np

CGNS_TO_LOC = {'BAR_2'  :'EdgeCenter',
               'TRI_3'  :'FaceCenter',
               'TETRA_4':'CellCenter'}
LOC_TO_CGNS = {'EdgeCenter':'BAR_2',
               'FaceCenter':'TRI_3',
               'CellCenter':'TETRA_4'}


def duplicate_vtx(zone, vtx_pl):
  coord_nodes = [PT.get_node_from_name(zone, name) for name in ['CoordinateX','CoordinateY','CoordinateZ']]
  for coord_n, coord in zip(coord_nodes, PT.Zone.coordinates(zone)):
    PT.set_value(coord_n, np.concatenate([coord, coord[vtx_pl-1]]))
  zone_dim = PT.get_value(zone)
  zone_dim[0][0] += vtx_pl.size
  PT.set_value(zone, zone_dim)


def remove_vtx(zone, vtx_pl):
  coord_nodes = [PT.get_node_from_name(zone, name) for name in ['CoordinateX','CoordinateY','CoordinateZ']]
  for coord_n, coord in zip(coord_nodes, PT.Zone.coordinates(zone)):
    PT.set_value(coord_n, np.delete(coord, vtx_pl-1))
  zone_dim = PT.get_value(zone)
  zone_dim[0][0] -= vtx_pl.size
  PT.set_value(zone, zone_dim)


def apply_periodicity_to_vtx(zone, vtx_pl, periodic):
  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  cx[vtx_pl-1], cy[vtx_pl-1], cz[vtx_pl-1] = np_utils.transform_cart_vectors(cx[vtx_pl-1],cy[vtx_pl-1],cz[vtx_pl-1], **periodic)
  PT.set_value(cx_n, cx)
  PT.set_value(cy_n, cy)
  PT.set_value(cz_n, cz)


def duplicate_flowsol_elts(zone, ids, location):
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)==location
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      da = PT.get_value(da_n)
      da_to_add = da[ids]
      PT.set_value(da_n, np.concatenate([da, da_to_add]))


def remove_flowsol_elts(zone, ids, loc):
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)==loc
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      da = PT.get_value(da_n)
      da = np.delete(da, ids)
      PT.set_value(da_n, da)


def apply_periodicity_to_flowsol(zone, ids, location, periodic):
  '''
  Apply periodicity to ids of vector fields in zone.
  '''
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)==location
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    # > Transform vector arrays : Get from maia.algo.transform.py
    data_names = [PT.get_name(data) for data in PT.iter_nodes_from_label(fs_n, "DataArray_t")]
    cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names)
    for basename in cartesian_vectors_basenames:
      vectors_n = [PT.get_node_from_name_and_label(fs_n, f"{basename}{c}", 'DataArray_t') for c in ['X', 'Y', 'Z']]
      vectors = [PT.get_value(n)[ids] for n in vectors_n]
      # Assume that vectors are position independant
      # Be careful, if coordinates vector needs to be transform, the translation is not applied !
      tr_vectors = np_utils.transform_cart_vectors(*vectors, rotation_center=periodic['rotation_center'], rotation_angle=periodic['rotation_angle'])
      for vector_n, tr_vector in zip(vectors_n, tr_vectors):
        vector = PT.get_value(vector_n)
        vector[ids] = tr_vector
        PT.set_value(vector_n, vector)
    

def elmt_pl_to_vtx_pl(zone, elt_pl, cgns_name):
  '''
  Return point_list of vertices describing elements tagged in `elt_pl`.
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
  ids = elt_pl - elt_offset
  conn_pl = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
  vtx_pl = np.unique(ec[conn_pl])

  return vtx_pl


def tag_elmt_owning_vtx(zone, vtx_pl, cgns_name, elt_full=False):
  '''
  Return the the point_list of elements that owns one or all of their vertices in the vertex point_list.
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n    = PT.get_node_from_predicate(zone, is_asked_elt)
  if elt_n is not None:
    n_elt      = PT.Element.Size(elt_n)
    elt_size   = PT.Element.NVtx(elt_n)
    elt_offset = PT.Element.Range(elt_n)[0]

    ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
    tag_vtx = np.isin(ec, vtx_pl) # True where vtx is
    if elt_full:
      tag_elt = np.logical_and.reduceat(tag_vtx, np.arange(0,n_elt*elt_size,elt_size)) # True when has vtx 
    else:
      tag_elt = np.logical_or .reduceat(tag_vtx, np.arange(0,n_elt*elt_size,elt_size)) # True when has vtx 
    gc_elt_pl = np.where(tag_elt)[0]+elt_offset # Which cells has vtx, TODO: add elt_offset to be a real pl
  else:
    gc_elt_pl = np.empty(0, dtype=np.int32)
  return gc_elt_pl


def is_elt_included(zone, src_pl, src_name, tgt_pl, tgt_name):
  '''
  Search which source element is a part of target elements.
  TODO: n_src*n_tgt algo, search better way to do it (NGon approach ?)
  '''
  # > Get source ec
  is_src_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)==src_name
  src_elt_n = PT.get_node_from_predicate(zone, is_src_elt)
  size_src_elt   = PT.Element.NVtx(src_elt_n)
  src_elt_offset = PT.Element.Range(src_elt_n)[0]
  src_ec_n       = PT.get_child_from_name(src_elt_n, 'ElementConnectivity')
  src_ec         = PT.get_value(src_ec_n)
  idx        = src_pl - src_elt_offset
  src_ec_pl  = np_utils.interweave_arrays([size_src_elt*idx+i_size for i_size in range(size_src_elt)])
  src_ec_elt = src_ec[src_ec_pl]
  n_src_elt  = src_pl.size

  # > Get target ec
  is_tgt_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)==tgt_name
  tgt_elt_n = PT.get_node_from_predicate(zone, is_tgt_elt)
  n_tgt_elt = PT.Element.Size(tgt_elt_n)
  size_tgt_elt   = PT.Element.NVtx(tgt_elt_n)
  tgt_elt_offset = PT.Element.Range(tgt_elt_n)[0]
  tgt_ec_n       = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec         = PT.get_value(tgt_ec_n)
  idx        = tgt_pl - tgt_elt_offset
  tgt_ec_pl  = np_utils.interweave_arrays([size_tgt_elt*idx+i_size for i_size in range(size_tgt_elt)])
  tgt_ec_elt = tgt_ec[tgt_ec_pl]
  n_tgt_elt  = tgt_pl.size

  mask = np.zeros(src_pl.size, dtype=np.int32)
  cdist_algo.find_tri_in_tetras(n_src_elt, n_tgt_elt, src_ec_elt, tgt_ec_elt, mask)
  
  return src_pl[mask.astype(bool)]


def update_elt_vtx_numbering(zone, old_to_new_vtx, cgns_name, elt_pl=None):
  '''
  Update element connectivity (partialy if `elt_pl` provided) according to the new vertices numbering described in `old_to_new_vtx`.
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  if elt_pl is None:
    ec = np.take(old_to_new_vtx, ec-1)
  else:
    elt_size   = PT.Element.NVtx(elt_n)
    elt_offset = PT.Element.Range(elt_n)[0]
    
    ids   = elt_pl - elt_offset
    ec_pl = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
    ec[ec_pl] = np.take(old_to_new_vtx, ec[ec_pl]-1)

  PT.set_value(ec_n, ec)


def remove_elts_from_pl(zone, elt_pl, cgns_name):
  '''
  Remove elements tagged in `elt_pl` by updating its ElementConnectivity and ElementRange nodes,
  as well as ElementRange nodes of elements with inferior dimension (assuming that element nodes are organized with decreasing dimension order).
  '''
  # > Get element information
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt      = PT.Element.Size(elt_n)
  elt_dim    = PT.Element.Dimension(elt_n)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)
  er_n = PT.get_child_from_name(elt_n, 'ElementRange')
  er   = PT.get_value(er_n)

  # > Updating element range and connectivity
  n_elt_to_rm = elt_pl.size
  pl_c  = -np.ones(n_elt_to_rm*elt_size, dtype=np.int32)
  for i_size in range(elt_size):
    pl_c[i_size::elt_size] = elt_size*(elt_pl-elt_offset)+i_size
  ec = np.delete(ec, pl_c)
  PT.set_value(ec_n, ec)

  er[1] = er[1]-n_elt_to_rm
  PT.set_value(er_n, er)

  # > Update BC PointList
  old_to_new_elt = np.arange(1, n_elt+1, dtype=np.int32)
  old_to_new_elt[elt_pl-elt_offset] = -1
  old_to_new_elt[np.where(old_to_new_elt!=-1)[0]] = np.arange(1, n_elt-elt_pl.size+1)

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl   = PT.get_value(bc_pl_n)[0]

    not_in_pl = np.isin(bc_pl, elt_pl, invert=True)
    new_bc_pl = old_to_new_elt[bc_pl[not_in_pl]-elt_offset]
    tag_invalid_elt = np.isin(new_bc_pl,-1, invert=True)
    new_bc_pl = new_bc_pl[tag_invalid_elt]
    new_bc_pl = new_bc_pl+elt_offset-1

    if new_bc_pl.size==0:
      PT.rm_child(zone_bc_n, bc_n)
    else:
      PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))

  # > Update element nodes with inferior dimension
  update_infdim_elts(zone, elt_dim, -n_elt_to_rm)


def update_infdim_elts(zone, elt_dim, offset):
  '''
  Go through all inferior dimension elements applying offset to their ElementRange.
  '''
  # > Updating offset others elements
  elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  for dim in range(elt_dim-1,0,-1):
    assert len(elts_per_dim[dim]) in [0,1]
    if len(elts_per_dim[dim])!=0:
      infdim_elt_n = elts_per_dim[dim][0]

      infdim_elt_range_n = PT.get_child_from_name(infdim_elt_n, 'ElementRange')
      infdim_elt_range = PT.get_value(infdim_elt_range_n)
      infdim_elt_range[0] = infdim_elt_range[0]+offset
      infdim_elt_range[1] = infdim_elt_range[1]+offset
      PT.set_value(infdim_elt_range_n, infdim_elt_range)

      infdim_elt_name = PT.Element.CGNSName(infdim_elt_n)
      is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                  PT.Subset.GridLocation(n)==CGNS_TO_LOC[infdim_elt_name]
      for elt_bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc):
        pl_n = PT.get_child_from_name(elt_bc_n, 'PointList')
        pl = PT.get_value(pl_n)
        PT.set_value(pl_n, pl+offset)


def merge_periodic_bc(zone, bc_names, vtx_tag, old_to_new_vtx_num, keep_original=False):
  '''
  Merge two similar BCs using a vtx numbering and a table describing how to transform vertices from first BC to second BC vertices.
  First BC can be kept using `keep_original` argument.
  '''
  n_vtx = PT.Zone.n_vtx(zone)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  pbc1_n      = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_loc    = PT.Subset.GridLocation(pbc1_n)
  pbc1_pl     = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  pbc1_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc1_pl, LOC_TO_CGNS[pbc1_loc])

  pbc2_n      = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_loc    = PT.Subset.GridLocation(pbc2_n)
  pbc2_pl     = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  pbc2_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc2_pl, LOC_TO_CGNS[pbc2_loc])

  old_vtx_num = old_to_new_vtx_num[0]
  new_vtx_num = old_to_new_vtx_num[1]

  pl1_tag = vtx_tag[pbc1_vtx_pl-1]
  sort_old = np.argsort(old_vtx_num)
  idx_pl1_tag_in_old = np.searchsorted(old_vtx_num, pl1_tag, sorter=sort_old)

  pl2_tag = vtx_tag[pbc2_vtx_pl-1]
  sort_pl2_tag = np.argsort(pl2_tag)
  idx_new_in_pl2_tag = np.searchsorted(pl2_tag, new_vtx_num, sorter=sort_pl2_tag)

  sources = pbc2_vtx_pl[sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]]
  targets = pbc1_vtx_pl

  old_to_new_vtx = np.arange(1, n_vtx+1, dtype=np.int32)
  old_to_new_vtx[sources-1] = -1
  old_to_new_vtx[np.where(old_to_new_vtx!=-1)[0]] = np.arange(1, n_vtx-sources.size+1)
  old_to_new_vtx[sources-1] = old_to_new_vtx[targets-1]
  
  if not keep_original:
    remove_elts_from_pl(zone, pbc1_pl, LOC_TO_CGNS[pbc2_loc])
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  remove_elts_from_pl(zone, pbc2_pl, LOC_TO_CGNS[pbc2_loc])
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TETRA_4')
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TRI_3')
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'BAR_2')

  n_vtx_to_rm = pbc2_vtx_pl.size
  remove_vtx(zone, pbc2_vtx_pl)

  # > Update Vertex BCs and GCs
  update_vtx_bnds(zone, old_to_new_vtx)

  # > Update flow_sol
  remove_flowsol_elts(zone, pbc2_vtx_pl-1, 'Vertex')

  return old_to_new_vtx


def update_vtx_bnds(zone, old_to_new_vtx):
  '''
  Update Vertex BCs and GCs according to the new vertices numbering described in `old_to_new_vtx`.
  '''

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  if zone_bc_n is not None:
    is_vtx_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)=='Vertex'
    for bc_n in PT.get_children_from_predicate(zone_bc_n, is_vtx_bc):
      bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
      bc_pl   = PT.get_value(bc_pl_n)[0]
      bc_pl   = np.take(old_to_new_vtx, bc_pl-1)
      assert (bc_pl!=-1).any()
      PT.set_value(bc_pl_n, bc_pl.reshape((1,-1), order='F'))

  zone_gc_n = PT.get_child_from_label(zone, 'ZoneGridConnectivity_t')
  if zone_gc_n is not None:
    is_vtx_gc = lambda n: PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t']  and\
                          PT.Subset.GridLocation(n)=='Vertex'
    for gc_n in PT.get_children_from_predicate(zone_gc_n, is_vtx_gc):
      gc_pl_n = PT.get_child_from_name(gc_n, 'PointList')
      gc_pl   = PT.get_value(gc_pl_n)[0]
      gc_pl   = np.take(old_to_new_vtx, gc_pl-1)
      assert (gc_pl!=-1).any()
      PT.set_value(gc_pl_n, gc_pl.reshape((1,-1), order='F'))

      gc_pld_n = PT.get_child_from_name(gc_n, 'PointListDonor')
      gc_pld   = PT.get_value(gc_pld_n)[0]
      gc_pld   = np.take(old_to_new_vtx, gc_pld-1)
      assert (gc_pld!=-1).any()
      PT.set_value(gc_pld_n, gc_pld.reshape((1,-1), order='F'))


def duplicate_elts(zone, elt_pl, elt_name, as_bc, elts_to_update, elt_duplicate_bcs=dict()):
  '''
  Duplicate elements tagged in `elt_pl` by updating its ElementConnectivity and ElementRange nodes,
  as well as ElementRange nodes of elements with inferior dimension (assuming that element nodes are organized with decreasing dimension order).
  Created elements can be tagged in a new BC_t through `as_bc` argument.
  Elements of other dimension touching vertices of duplicated elements can be updating with new created vertices through `elts_to_update` argument.
  BCs fully described by duplicated element vertices can be duplicated as well using `elt_duplicate_bcs` argument (lineic BCs in general).
  '''
  # > Get element informations
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                             PT.Element.CGNSName(n)==elt_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt      = PT.Element.Size(elt_n)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  elt_dim    = PT.Element.Dimension(elt_n)


  # > Add duplicated vertex
  n_vtx = PT.Zone.n_vtx(zone)
  elt_vtx_pl   = elmt_pl_to_vtx_pl(zone, elt_pl, elt_name)
  n_vtx_to_add = elt_vtx_pl.size
  duplicate_vtx(zone, elt_vtx_pl)

  new_vtx_pl  = np.arange(n_vtx, n_vtx+n_vtx_to_add)+1
  new_vtx_num = [elt_vtx_pl,new_vtx_pl]
  old_to_new_vtx = np.arange(1, n_vtx+1)
  old_to_new_vtx[new_vtx_num[0]-1] = new_vtx_num[1]
  sort_vtx_num = np.argsort(new_vtx_num[0])
  
  duplicate_flowsol_elts(zone, elt_vtx_pl-1, 'Vertex')

  # > Add duplicated elements
  n_elt_to_add = elt_pl.size
  
  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  er[1] = er[1]+n_elt_to_add

  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = elt_pl - elt_offset
  ec_pl  = np_utils.interweave_arrays([elt_size*pl+i_size for i_size in range(elt_size)])
  elt_ec = ec[ec_pl]
  idx_vtx_num = np.searchsorted(new_vtx_num[0], elt_ec, sorter=sort_vtx_num)
  elt_ec = new_vtx_num[1][sort_vtx_num[idx_vtx_num]]

  PT.set_value(er_n, er)
  PT.set_value(ec_n, np.concatenate([ec, elt_ec]))

  update_infdim_elts(zone, elt_dim, n_elt_to_add)

  # > Create associated BC if asked
  if as_bc is not None:
    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    new_elt_pl = np.arange(n_elt, n_elt+n_elt_to_add, dtype=np.int32) + elt_offset
    PT.new_BC(name=as_bc, 
              type='FamilySpecified',
              point_list=new_elt_pl.reshape((1,-1), order='F'),
              loc=CGNS_TO_LOC[elt_name],
              family='PERIODIC',
              parent=zone_bc_n)

  # > Duplicate twin BCs
  new_ec = list()
  twin_elt_bc_pl = dict()
  twin_elt_bc_pl['BAR_2'] = list()
  for elt_name, duplicate_bcs in elt_duplicate_bcs.items():
    # > Get element infos
    is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                             PT.Element.CGNSName(n)==elt_name
    elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
    n_elt      = PT.Element.Size(elt_n)
    elt_size   = PT.Element.NVtx(elt_n)
    elt_offset = PT.Element.Range(elt_n)[0]

    ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
    ec    = PT.get_value(ec_n)

    n_new_elt = 0
    for matching_bcs in duplicate_bcs:
      bc_n = PT.get_child_from_name(zone_bc_n, matching_bcs[0])
      bc_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
      twin_elt_bc_pl['BAR_2'].append(bc_pl)

      ids = bc_pl - elt_offset
      ec_pl = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
      bc_ec = ec[ec_pl]
      new_bc_ec = np.take(old_to_new_vtx, bc_ec-1)
      new_ec.append(new_bc_ec)

      new_bc_pl = np.arange(n_elt+n_new_elt, n_elt+n_new_elt+bc_pl.size) + elt_offset
      new_bc_n = PT.new_BC(name=matching_bcs[1],
                           type='FamilySpecified',
                           point_list=new_bc_pl.reshape((1,-1), order='F'),
                           loc=CGNS_TO_LOC[elt_name],
                           parent=zone_bc_n)
      bc_fam_n = PT.get_child_from_label(bc_n, 'FamilyName_t')
      if bc_fam_n is not None:
        PT.new_node('FamilyName', value=PT.get_value(bc_fam_n), label='FamilyName_t', parent=new_bc_n)
      n_new_elt += bc_pl.size

    # > Update elt node
    ec = np.concatenate([ec]+new_ec)
    PT.set_value(ec_n, ec)
    er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
    er    = PT.get_value(er_n)
    er[1]+= n_new_elt
    PT.set_value(er_n, er)


  # > Update vtx numbering of elements in patch to separate patch
  cell_pl = elts_to_update['TETRA_4']
  face_pl = elts_to_update['TRI_3']
  line_pl = elts_to_update['BAR_2']+n_elt_to_add # Attention au décalage de la PL 

  # > Exclude constraint surf or both will move
  tag_face = np.isin(face_pl, elt_pl, invert=True)
  face_pl  = face_pl[tag_face]
  bc_line_pl = np.concatenate(twin_elt_bc_pl['BAR_2']) if len(twin_elt_bc_pl['BAR_2'])!=0 else np.empty(0, dtype=np.int32)
  tag_line = np.isin(line_pl, bc_line_pl, invert=True)
  line_pl  = line_pl[tag_line]

  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TETRA_4', elt_pl=cell_pl)
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TRI_3'  , elt_pl=face_pl)
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'BAR_2'  , elt_pl=line_pl)

  return new_vtx_num

    
def find_matching_bcs(zone, src_pl, tgt_pl, src_tgt_vtx, cgns_name):
  '''
  Find pairs of twin BCs (lineic in general) using a matching vtx table.
  '''
  n_vtx = PT.Zone.n_vtx(zone)

  # > Find BCs described by pls
  bc_nodes = [list(),list()]
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc):
    bc_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
    for i_side, elt_pl in enumerate([src_pl, tgt_pl]):
      mask = np.isin(bc_pl, elt_pl, assume_unique=True) # See if other np.isin can have assumer_unique
      if np.logical_and.reduce(mask):
        bc_nodes[i_side].append(bc_n)

  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  elt_offset = PT.Element.Range(elt_n)[0]
  elt_size = PT.Element.NVtx(elt_n)

  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)
  
  old_to_new_vtx = np.arange(1, n_vtx+1, dtype=np.int32)
  old_to_new_vtx[src_tgt_vtx[0]-1] = src_tgt_vtx[1] # Normally, elements has no vtx in common
  
  # > Go through BCs described by join vertices and find pairs
  matching_bcs = list()
  for src_bc_n in bc_nodes[0]:
    src_bc_pl = PT.get_value(PT.Subset.getPatch(src_bc_n))[0]
    pl = src_bc_pl - elt_offset
    ec_pl = np_utils.interweave_arrays([elt_size*pl+i_size for i_size in range(elt_size)])
    as_tgt_ec = np.take(old_to_new_vtx, ec[ec_pl]-1)
    for tgt_bc_n in bc_nodes[1]:
      tgt_bc_pl = PT.get_value(PT.Subset.getPatch(tgt_bc_n))[0]
      pl = tgt_bc_pl - elt_offset
      ec_pl = np_utils.interweave_arrays([elt_size*pl+i_size for i_size in range(elt_size)])
      tgt_ec = np.take(old_to_new_vtx, ec[ec_pl]-1)
      mask = np.isin(as_tgt_ec, tgt_ec)
      if np.logical_and.reduce(mask):
        matching_bcs.append([PT.get_name(tgt_bc_n), PT.get_name(src_bc_n)])

  return matching_bcs


def add_undefined_faces(zone, elt_pl, elt_name, vtx_pl, tgt_elt_name):
  '''
  Add faces (TRI_3) in mesh which are face from cells that are not touching the join here described by vtx point_list.
  Check that created face are not already descibed in BCs, or described by two cells (it is an internal face in this case).
  '''
  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==elt_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  ec_n       = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec         = PT.get_value(ec_n)
  n_elt_to_add = elt_pl.size

  is_tgt_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==tgt_elt_name
  tgt_elt_n      = PT.get_node_from_predicate(zone, is_tgt_elt)
  dim_tgt_elt    = PT.Element.Dimension(tgt_elt_n)
  size_tgt_elt   = PT.Element.NVtx(tgt_elt_n)
  tgt_elt_offset = PT.Element.Range(tgt_elt_n)[0]
  tgt_ec_n       = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec         = PT.get_value(tgt_ec_n)

  # > Get elts connectivity
  idx    = elt_pl - elt_offset
  ec_pl  = np_utils.interweave_arrays([elt_size*idx+i_size for i_size in range(elt_size)])
  ec_elt = ec[ec_pl]

  # > Get BCs of tgt dimension to get their vtx ids 
  is_tgt_elt_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)==CGNS_TO_LOC[tgt_elt_name]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_bc_elt = 0
  bc_ecs = list()
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_tgt_elt_bc):
    bc_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0] 
    ids   = bc_pl - tgt_elt_offset
    bc_ec_pl  = np_utils.interweave_arrays([size_tgt_elt*ids+i_size for i_size in range(size_tgt_elt)])
    ec_bc = tgt_ec[bc_ec_pl]
    n_bc_elt += ids.size
    bc_ecs.append(ec_bc)


  # > Find cells with 3 vertices not in GC
  tag_elt = np.isin(ec_elt, vtx_pl, invert=True)
  tag_elt = np.add.reduceat(tag_elt.astype(np.int32), np.arange(0,n_elt_to_add*elt_size,elt_size)) # True when has vtx 
  elt_pl = elt_pl[np.where(tag_elt==elt_size-1)[0]]
  n_elt_to_add = elt_pl.size

  # > Get elts connectivity
  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = elt_pl -1
  ec_pl  = np_utils.interweave_arrays([elt_size*pl+i_size for i_size in range(elt_size)])
  tgt_elt_ec = ec[ec_pl]


  # > Face reconstruction configurations from tetrahedras
  conf0 = np.array([0, 1, 2], dtype=np.int32)
  # conf1 = np.array([0, 1, 3], dtype=np.int32)
  conf2 = np.array([0, 2, 3], dtype=np.int32)
  # conf3 = np.array([1, 2, 3], dtype=np.int32)
  
  conf0t = np.array([0, 2, 1], dtype=np.int32)
  # conf1t = np.array([0, 1, 3], dtype=np.int32)
  conf2t = np.array([0, 2, 1], dtype=np.int32)
  # conf3t = np.array([1, 2, 3], dtype=np.int32)

  tag_elt = np.isin(tgt_elt_ec, vtx_pl, invert=True)
  tag_elt_rshp = tag_elt.reshape(n_elt_to_add,elt_size)
  tag_eltm1 = np.where(tag_elt_rshp)
  tag_eltm1_rshp = tag_eltm1[1].reshape(n_elt_to_add,elt_size-1)

  tgt_elt_ec = tgt_elt_ec[tag_elt].reshape(n_elt_to_add,elt_size-1)
  for conf, conft in zip([conf0,conf2], [conf0t,conf2t]):
    tag_conf = np.where((tag_eltm1_rshp==conf).all(1))[0]
    tgt_elt_ec_cp = tgt_elt_ec[tag_conf]
    tgt_elt_ec_cp = tgt_elt_ec_cp[:,conft]
    tgt_elt_ec[tag_conf] = tgt_elt_ec_cp
  tgt_elt_ec = tgt_elt_ec.reshape(n_elt_to_add*(elt_size-1))


  # > Find faces not already defined in BCs or duplicated
  bc_ec    = np.concatenate(bc_ecs)
  tmp_mask = np.full(n_elt_to_add+n_bc_elt, 1, dtype=np.int32)
  tmp_ec   = np.concatenate([tgt_elt_ec, bc_ec])
  cdist_algo.find_duplicate_elt(n_elt_to_add+n_bc_elt, size_tgt_elt, tmp_ec, tmp_mask)
  elt_ids = np.where(tmp_mask[0:n_elt_to_add]==1)[0] # Get only tri which are not in BCs
  n_elt_to_add = elt_ids.size
  ec_pl = np_utils.interweave_arrays([size_tgt_elt*elt_ids+i_size for i_size in range(size_tgt_elt)])
  tgt_elt_ec = tgt_elt_ec[ec_pl]


  # Update target element
  tgt_ec_n   = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec     = PT.get_value(tgt_ec_n)
  PT.set_value(tgt_ec_n, np.concatenate([tgt_ec, tgt_elt_ec]))

  tgt_er_n  = PT.get_child_from_name(tgt_elt_n, 'ElementRange')
  tgt_er    = PT.get_value(tgt_er_n)
  new_tgt_elt_pl = np.arange(tgt_er[1], tgt_er[1]+n_elt_to_add)+elt_offset
  tgt_er[1] = tgt_er[1]+n_elt_to_add
  PT.set_value(tgt_er_n, tgt_er)

  update_infdim_elts(zone, dim_tgt_elt, n_elt_to_add)

  return new_tgt_elt_pl


def convert_vtx_gcs_as_face_bcs(tree, gc_paths):
  '''
  Convert Vertex GCs as FaceCenter BCs for feflo.
  '''
  zone = PT.get_node_from_label(tree, 'Zone_t')
  zone_bc_n = PT.get_node_from_label(tree, 'ZoneBC_t')

  # > Get TRI_3 element infos
  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)=='TRI_3'
  elt_n      = PT.get_node_from_predicate(zone, is_tri_elt)
  n_elt      = PT.Element.Size(elt_n)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  ec_n       = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec         = PT.get_value(ec_n)

  for i_side, side_gc_paths in enumerate(gc_paths):
    for i_gc, gc_path in enumerate(side_gc_paths):
      # > Get GCs infos
      gc_n    = PT.get_node_from_path(tree, gc_path)
      gc_name = PT.get_name(gc_n)
      gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
      gc_loc  = PT.Subset.GridLocation(gc_n)
      assert gc_loc=='Vertex', ''

      # > Search faces described by gc vtx
      tag_elt = np.isin(ec, gc_pl)
      tag_elt = np.logical_and.reduceat(tag_elt, np.arange(0,n_elt*elt_size,elt_size))
      bc_pl   = np.where(tag_elt)[0]+elt_offset

      bc_n = PT.new_BC(name=gc_name,
                       type='FamilySpecified',
                       point_list=bc_pl.reshape((1,-1), order='F'),
                       loc='FaceCenter',
                       family='GCS',
                       parent=zone_bc_n)


def deplace_periodic_patch(tree, gc_paths, periodic_values):
  '''
  Use `gc_paths` and their associated `periodic_values` to deplace a range of cell touching a GC next to the twin GC (for each pair of GCs),
  using GCs pl and pls to merge the two domains.
  This function is not working on domains where some vertices are shared by GCs (from different pairs or not).
  '''
  base = PT.get_child_from_label(tree, 'CGNSBase_t')

  zones = PT.get_nodes_from_label(base, 'Zone_t')
  assert len(zones)==1
  zone = zones[0]

  # > Add GCs as BCs (paraview visu mainly)
  PT.new_Family('PERIODIC', parent=base)
  PT.new_Family('GCS', parent=base)

  n_periodicity = len(gc_paths[0])
  new_vtx_nums = list()
  to_constrain_bcs = list()
  matching_bcs   = [dict() for i_per in range(n_periodicity)]
  for i_per in range(n_periodicity):

    gc_vtx_n = PT.get_node_from_path(tree, gc_paths[0][i_per])
    gc_vtx_pl  = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointList'     ))[0]
    gc_vtx_pld = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointListDonor'))[0]

    cell_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, 'TETRA_4', elt_full=False)
    face_pl = add_undefined_faces(zone, cell_pl, 'TETRA_4', gc_vtx_pld, 'TRI_3', )
    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4')

    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    PT.new_BC(name=cell_bc_name, 
              type='FamilySpecified',
              point_list=cell_pl.reshape((1,-1), order='F'),
              loc='CellCenter',
              family='PERIODIC',
              parent=zone_bc_n)
    face_bc_name = f'tri_3_constraint_{i_per}'
    PT.new_BC(name=face_bc_name, 
              type='FamilySpecified',
              point_list=face_pl.reshape((1,-1), order='F'),
              loc='FaceCenter',
              family='PERIODIC',
              parent=zone_bc_n)
    to_constrain_bcs.append(face_bc_name)

    maia.io.write_tree(tree, 'OUTPUT/internal_surface.cgns')

    # > Removing lines defined on join because they surely has their periodic on the other side
    # > Find BCs on GCs that will be deleted because they have their periodic twin
    # > For now only fully described BCs will be treated
    matching_bcs[i_per] = dict()
    bar_to_rm_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, 'BAR_2', elt_full=True)
    bar_twins_pl = tag_elmt_owning_vtx(zone, gc_vtx_pl,  'BAR_2', elt_full=True)
    matching_bcs[i_per]['BAR_2'] = find_matching_bcs(zone, bar_to_rm_pl, bar_twins_pl, [gc_vtx_pld, gc_vtx_pl], 'BAR_2')
    remove_elts_from_pl(zone, bar_to_rm_pl, 'BAR_2')


    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_pl
    to_update_face_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'TRI_3', elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'BAR_2', elt_full=True)

    # > Ambiguous faces that contains all vtx but are not included in patch cells can be removed by searching
    # > faces that contains all vtx froms cells that are not in patch
    to_update_face_pl = is_elt_included(zone, to_update_face_pl, 'TRI_3', cell_pl, 'TETRA_4')

    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}
  
    face_bc_name = f'tri_3_periodic_{i_per}'
    new_vtx_num = duplicate_elts(zone, face_pl, 'TRI_3', f'tri_3_periodic_{i_per}', elts_to_update)
    to_constrain_bcs.append(face_bc_name)

    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4')
    apply_periodicity_to_vtx(zone, vtx_pl, periodic_values[1][i_per])
    apply_periodicity_to_flowsol(zone, vtx_pl-1, 'Vertex', periodic_values[1][i_per])

    maia.io.write_tree(tree, 'OUTPUT/deplaced.cgns')

    n_vtx = PT.Zone.n_vtx(zone)
    bc_name1 = gc_paths[0][i_per].split('/')[-1]
    bc_name2 = gc_paths[1][i_per].split('/')[-1]
    vtx_match_num = [gc_vtx_pl, gc_vtx_pld]
    vtx_tag = np.arange(1, n_vtx+1, dtype=np.int32)
    old_to_new_vtx = merge_periodic_bc(zone, (bc_name1, bc_name2), vtx_tag, vtx_match_num, keep_original=True)
    
    for i_previous_per in range(0, i_per):
      new_vtx_nums[i_previous_per][0] = np.take(old_to_new_vtx, new_vtx_nums[i_previous_per][0]-1)
      new_vtx_nums[i_previous_per][1] = np.take(old_to_new_vtx, new_vtx_nums[i_previous_per][1]-1)
    new_vtx_num[0] = np.take(old_to_new_vtx, new_vtx_num[0]-1)
    new_vtx_num[1] = np.take(old_to_new_vtx, new_vtx_num[1]-1)
    new_vtx_nums.append(new_vtx_num)

    # > Set Vertex BC to preserve join infos
    pl_constraint = new_vtx_num[0].reshape((1,-1), order='F')
    PT.new_BC(name=f'vtx_constraint_{i_per}',
              type='FamilySpecified',
              point_list=pl_constraint,
              loc='Vertex',
              family='PERIODIC',
              parent=zone_bc_n)
    pl_periodic = new_vtx_num[1].reshape((1,-1), order='F')
    PT.new_BC(name=f'vtx_periodic_{i_per}',
              type='FamilySpecified',
              point_list=pl_periodic,
              loc='Vertex',
              family='PERIODIC',
              parent=zone_bc_n)

  return new_vtx_nums, to_constrain_bcs, matching_bcs


def retrieve_initial_domain(tree, gc_paths, periodic_values, new_vtx_num, bcs_to_retrieve):
  '''
  Use `gc_paths` and their associated `periodic_values` to deplace the adapted range of cells that have been adapted to its initial position,
  using `new_vtx_num` match info to merge the two domains.
  Twin BCs defined in GCs that have been deleted while using `deplace_periodic_patch` can be retrieved thanks to `bcs_to_retrieve`.
  '''
  n_periodicity = len(gc_paths[0])

  base = PT.get_child_from_label(tree, 'CGNSBase_t')
  is_3d = PT.get_value(base)[0]==3

  zone = PT.get_child_from_label(base, 'Zone_t')
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  cell_elt_name = 'TETRA_4' if is_3d else 'TRI_3'
  face_elt_name = 'TRI_3'   if is_3d else 'BAR_2'
  edge_elt_name = 'BAR_2'   if is_3d else  None

  # > Removing old periodic patch
  for i_per in range(n_periodicity-1, -1, -1):
  
    # > Duplicate GC surface and update element connectivities in the patch
    # > Here we can take all elements in periodic patch because its good from previous step
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    cell_bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_bc_pl = PT.get_value(PT.Subset.getPatch(cell_bc_n))[0]
    vtx_pl = elmt_pl_to_vtx_pl(zone, cell_bc_pl, 'TETRA_4')

    still_here_gc_name  = gc_paths[0][i_per].split('/')[-1]
    to_retrieve_gc_name = gc_paths[1][i_per].split('/')[-1]
    bc_n = PT.get_child_from_name(zone_bc_n, still_here_gc_name)
    face_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]

    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_bc_pl
    to_update_face_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'TRI_3', elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'BAR_2', elt_full=True)
    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}

    _ = duplicate_elts(zone, face_pl, 'TRI_3', to_retrieve_gc_name, elts_to_update, elt_duplicate_bcs=bcs_to_retrieve[i_per])
    # maia.io.write_tree(tree, f'OUTPUT/adapted_and_duplicated_{i_per}.cgns')

    # > Deplace periodic patch to retrieve initial domain
    # > vtx_pl is updated because has changed with surface duplication
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4')
    apply_periodicity_to_vtx(zone, vtx_pl, periodic_values[0][i_per])
    apply_periodicity_to_flowsol(zone, vtx_pl-1, 'Vertex', periodic_values[0][i_per])
    # maia.io.write_tree(tree, f'OUTPUT/adapted_and_deplaced_{i_per}.cgns')

    # > Merge two constraint surfaces
    vtx_tag_n = PT.get_node_from_name(zone, 'vtx_tag')
    vtx_tag   = PT.get_value(vtx_tag_n)
    _ = merge_periodic_bc(zone,
                      [f'{face_elt_name.lower()}_constraint_{i_per}', f'{face_elt_name.lower()}_periodic_{i_per}'],
                      vtx_tag,
                      new_vtx_num[i_per])

  rm_feflo_added_elt(zone)


def rm_feflo_added_elt(zone):
  '''
  Remove BCs that are created by mesh adaptation with feflo.
  If BC location is not CellCenter the elements tagged in BCs are removed. 
  '''
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  feflo_bcs = PT.get_children_from_name(zone_bc_n, 'feflo_*')

  for bc_n in feflo_bcs:
    if PT.Subset.GridLocation(bc_n)=='CellCenter':
      PT.rm_child(zone_bc_n, bc_n)
    else:
      bc_loc = PT.Subset.GridLocation(bc_n)
      bc_pl  = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
      remove_elts_from_pl(zone, bc_pl, LOC_TO_CGNS[bc_loc])
