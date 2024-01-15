import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
import maia.transfer.protocols as EP
import maia.transfer.utils as te_utils
from   maia.utils  import np_utils, py_utils, par_utils
from   maia.algo.dist import transform as dist_transform
from   maia.algo.dist.subset_tools import vtx_ids_to_face_ids
from   maia.algo.dist.remove_element import remove_elts_from_pl
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np

import Pypdm.Pypdm as PDM

DIM_TO_LOC = {0:'Vertex',
              1:'EdgeCenter',
              2:'FaceCenter',
              3:'CellCenter'}
CGNS_TO_LOC = {'BAR_2'  :'EdgeCenter',
               'TRI_3'  :'FaceCenter',
               'TETRA_4':'CellCenter'}
LOC_TO_CGNS = {'EdgeCenter':'BAR_2',
               'FaceCenter':'TRI_3',
               'CellCenter':'TETRA_4'}


def duplicate_vtx(zone, vtx_pl, comm):
  '''
  Duplicate vtx tagged in `vtx_pl` from a distributed zone.
    - Vertex distribution must be updated outside.
    - What about zone_dim ?
  Note : if an id appear twice in vtx_pl, it will be duplicated twice
  '''
  distri_n = PT.maia.getDistribution(zone, 'Vertex')
  distri   = PT.get_value(distri_n)
  
  coords_keys = ['CoordinateX', 'CoordinateY', 'CoordinateZ']
  coord_nodes = {key: PT.get_node_from_name(zone, key) for key in coords_keys}
  coords = {key: PT.get_value(node) for key, node in coord_nodes.items()}
  new_coords = EP.block_to_part(coords, distri, [vtx_pl], comm)
  for key in coords_keys:
    PT.set_value(coord_nodes[key], np.concatenate([coords[key], new_coords[key][0]]))

  # > Update zone information
  n_added_vtx = comm.allreduce(vtx_pl.size, op=MPI.SUM)
  zone_dim = PT.get_value(zone)
  zone_dim[0][0] += n_added_vtx


def remove_vtx(zone, vtx_pl, comm):
  '''
  Remove vtx tagged in `vtx_pl` from a distributed zone.
    - Vertex distribution must be updated outside.
    - What about zone_dim ?
  '''
  distri_n = PT.maia.getDistribution(zone, 'Vertex')
  distri   = PT.get_value(distri_n)
  
  ptb = EP.PartToBlock(distri, [vtx_pl], comm)
  ids = ptb.getBlockGnumCopy()-distri[0]-1
  
  coord_nodes = [PT.get_node_from_name(zone, name) for name in ['CoordinateX','CoordinateY','CoordinateZ']]
  for coord_n, coord in zip(coord_nodes, PT.Zone.coordinates(zone)):
    PT.set_value(coord_n, np.delete(coord, ids))

  # > Update zone information
  n_rmvd_vtx = comm.allreduce(vtx_pl.size, op=MPI.SUM)
  zone_dim = PT.get_value(zone)
  zone_dim[0][0] -= n_rmvd_vtx

def duplicate_flowsol_elts(zone, ids, location, comm):
  '''
  Duplicate flowsol values tagged in `ids` from a distributed zone. FlowSol distribution must be updated outside.
  '''
  is_loc_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)==location
  for fs_n in PT.get_children_from_predicate(zone, is_loc_fs):
    distri = te_utils.get_subset_distribution(zone, fs_n)

    arrays_n = PT.get_children_from_label(fs_n, 'DataArray_t')
    arrays = {PT.get_name(array_n) : PT.get_value(array_n) for array_n in arrays_n}
    new_arrays = EP.block_to_part(arrays, distri, [ids+1], comm)
    for array_n in arrays_n:
      key = PT.get_name(array_n)
      PT.set_value(array_n, np.concatenate([arrays[key], new_arrays[key][0]]))


def remove_flowsol_elts(zone, ids, location, comm):
  '''
  Remove flowsol values tagged in `ids` from a distributed zone. FlowSol distribution must be updated outside.
  '''
  is_loc_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)==location
  for fs_n in PT.get_children_from_predicate(zone, is_loc_fs):
    distri = te_utils.get_subset_distribution(zone, fs_n)
    
    ptb = EP.PartToBlock(distri, [ids+1], comm)
    ids = ptb.getBlockGnumCopy()-distri[0]-1

    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      da = PT.get_value(da_n)
      PT.set_value(da_n, np.delete(da,ids))


def elmt_pl_to_vtx_pl(zone, elt_pl, cgns_name, comm):
  '''
  Return point_list of vertices describing elements tagged in `elt_pl`.

  TODO : c'est anormal de demander un nom cgns (ex. TRI_3) et de ne récupérer 1
  seul élément. Soit il faut récupérer tous les éléments TRI_3, soit il faut 
  donner en entrée un nom de noeud spécifique et ne récupérer que ce noeud.
  Idem pr function d'en dessous
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  distri     = PT.maia.getDistribution(elt_n, 'Element')[1]

  ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
  ids = elt_pl - elt_offset + 1
  
  _, all_vtx = EP.block_to_part_strided(elt_size, ec, distri, [ids], comm)
  return np.unique(all_vtx)


def tag_elmt_owning_vtx(zone, vtx_pl, cgns_name, comm, elt_full=False):
  '''
  Return the point_list of elements that owns one or all of their vertices in the vertex point_list.
  Important : elt_pl is returned as a distributed array, w/o any assumption on the holding
  rank : vertices given by a rank can spawn an elt_idx on a other rank.
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  if elt_n is not None:
    elt_offset = PT.Element.Range(elt_n)[0]
    gc_elt_pl  = vtx_ids_to_face_ids(vtx_pl, elt_n, comm, elt_full)+elt_offset-1
  else:
    gc_elt_pl = np.empty(0, dtype=np.int32)
  return gc_elt_pl


def is_elt_included(zone, src_pl, src_name, tgt_pl, tgt_name):
  '''
  Search which source element is a part of target elements.
  TODO: parallel ?
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

  
  tgt_face_vtx_idx, tgt_face_vtx = PDM.decompose_std_elmt_faces(PDM._PDM_MESH_NODAL_TETRA4, tgt_ec_elt)
  tmp_ec = np.concatenate([src_ec_elt, tgt_face_vtx])
  mask = np_utils.is_unique_strided(tmp_ec, size_src_elt, method='hash')
  mask = ~mask[0:n_src_elt]

  return src_pl[mask]


def update_elt_vtx_numbering(zone, old_to_new_vtx, cgns_name, elt_pl=None):
  '''
  Update element connectivity (partialy if `elt_pl` provided) according to the new vertices numbering described in `old_to_new_vtx`.
  TODO: parallel
  '''
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  if elt_n is not None:
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


def apply_offset_to_elts(zone, offset, min_range):
  '''
  Go through all elements with ElementRange>min_range, applying offset to their ElementRange.
  '''
  # > Add offset to elements with ElementRange>min_range
  treated_bcs = list()
  elt_nodes = PT.Zone.get_ordered_elements(zone)
  for elt_n in elt_nodes:
    if PT.Element.Range(elt_n)[0]>min_range:
      elt_dim    = PT.Element.Dimension(elt_n)

      elt_range_n  = PT.get_child_from_name(elt_n, 'ElementRange')
      elt_range    = PT.get_value(elt_range_n)
      elt_range[0] = elt_range[0]+offset
      elt_range[1] = elt_range[1]+offset
      PT.set_value(elt_range_n, elt_range)

  # > Treating all BCs outside of elts because if not elt of dim of BC,
  #   it wont be treated.
  for bc_n in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
    if PT.get_name(bc_n) not in treated_bcs:
      pl_n = PT.get_child_from_name(bc_n, 'PointList')
      pl   = PT.get_value(pl_n)
      pl[min_range<pl] += offset
      PT.set_value(pl_n, pl)
      treated_bcs.append(PT.get_name(bc_n))

def merge_periodic_bc(zone, bc_names, vtx_tag, old_to_new_vtx_num, comm, keep_original=False):
  '''
  Merge two similar BCs using a vtx numbering and a table describing how to transform vertices from first BC to second BC vertices.
  First BC can be kept using `keep_original` argument.
  TODO: sortir le double searchsorted, que deplace_periodic_patch soit moins cher, transformer merge_zone U_elmt?
  '''
  n_vtx = PT.Zone.n_vtx(zone)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  # TODO: directement choper les GCs
  pbc1_n      = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_loc    = PT.Subset.GridLocation(pbc1_n)
  pbc1_pl     = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  pbc1_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc1_pl, LOC_TO_CGNS[pbc1_loc], MPI.COMM_SELF)

  pbc2_n      = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_loc    = PT.Subset.GridLocation(pbc2_n)
  pbc2_pl     = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  pbc2_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc2_pl, LOC_TO_CGNS[pbc2_loc], MPI.COMM_SELF)

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
  
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==LOC_TO_CGNS[pbc2_loc]
  elt_n = PT.get_child_from_predicate(zone, is_asked_elt)
  if not keep_original:
    remove_elts_from_pl(zone, elt_n, pbc1_pl, comm)
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  remove_elts_from_pl(zone, elt_n, pbc2_pl, comm)
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TETRA_4')
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'TRI_3')
  update_elt_vtx_numbering(zone, old_to_new_vtx, 'BAR_2')

  n_vtx_to_rm = pbc2_vtx_pl.size
  remove_vtx(zone, pbc2_vtx_pl, MPI.COMM_SELF)

  # > Update Vertex BCs and GCs
  update_vtx_bnds(zone, old_to_new_vtx)

  # > Update flow_sol
  remove_flowsol_elts(zone, pbc2_vtx_pl-1, 'Vertex', MPI.COMM_SELF)
  
  # > Update distribution
  vtx_distrib_n  = PT.maia.getDistribution(zone, distri_name='Vertex')
  vtx_distrib    = PT.get_value(vtx_distrib_n)
  vtx_distrib[1]-= n_vtx_to_rm
  vtx_distrib[2]-= n_vtx_to_rm
  PT.set_value(vtx_distrib_n, vtx_distrib)
  
  return old_to_new_vtx


def update_vtx_bnds(zone, old_to_new_vtx):
  '''
  Update Vertex BCs and GCs according to the new vertices numbering described in `old_to_new_vtx`.
  TODO: parallel + predicates
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


def duplicate_elts(zone, elt_pl, elt_name, as_bc, elts_to_update, comm, elt_duplicate_bcs=dict()):
  '''
  Duplicate elements tagged in `elt_pl` by updating its ElementConnectivity and ElementRange nodes,
  as well as ElementRange nodes of elements with inferior dimension (assuming that element nodes are organized with decreasing dimension order).
  Created elements can be tagged in a new BC_t through `as_bc` argument.
  Elements of other dimension touching vertices of duplicated elements can be updating with new created vertices through `elts_to_update` argument.
  BCs fully described by duplicated element vertices can be duplicated as well using `elt_duplicate_bcs` argument (lineic BCs in general).
  TODO: parallel ?
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
  elt_vtx_pl   = elmt_pl_to_vtx_pl(zone, elt_pl, elt_name, MPI.COMM_SELF)
  n_vtx_to_add = elt_vtx_pl.size
  duplicate_vtx(zone, elt_vtx_pl, MPI.COMM_SELF)


  new_vtx_pl  = np.arange(n_vtx, n_vtx+n_vtx_to_add)+1
  new_vtx_num = [elt_vtx_pl,new_vtx_pl]
  old_to_new_vtx = np.arange(1, n_vtx+1)
  old_to_new_vtx[new_vtx_num[0]-1] = new_vtx_num[1]
  sort_vtx_num = np.argsort(new_vtx_num[0])
  
  duplicate_flowsol_elts(zone, elt_vtx_pl-1, 'Vertex', MPI.COMM_SELF)
  
  # > Update distribution
  vtx_distrib_n  = PT.maia.getDistribution(zone, distri_name='Vertex')
  vtx_distrib    = PT.get_value(vtx_distrib_n)
  vtx_distrib[1]+= n_vtx_to_add
  vtx_distrib[2]+= n_vtx_to_add
  PT.set_value(vtx_distrib_n, vtx_distrib)

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

  # > Update distribution
  elt_distrib_n  = PT.maia.getDistribution(elt_n, distri_name='Element')
  elt_distrib    = PT.get_value(elt_distrib_n)
  elt_distrib[1]+= n_elt_to_add
  elt_distrib[2]+= n_elt_to_add
  PT.set_value(elt_distrib_n, elt_distrib)

  apply_offset_to_elts(zone, n_elt_to_add, er[1]-n_elt_to_add)

  # > Create associated BC if asked
  if as_bc is not None:
    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    new_elt_pl = np.arange(n_elt, n_elt+n_elt_to_add, dtype=np.int32) + elt_offset
    new_bc_distrib = par_utils.dn_to_distribution(new_elt_pl.size, comm)
    bc_n = PT.new_BC(name=as_bc, 
                     type='FamilySpecified',
                     point_list=new_elt_pl.reshape((1,-1), order='F'),
                     loc=DIM_TO_LOC[elt_dim],
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':new_bc_distrib}, parent=bc_n)

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
    elt_dim    = PT.Element.Dimension(elt_n)

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
      new_bc_distrib = par_utils.dn_to_distribution(new_bc_pl.size, comm)
      new_bc_n = PT.new_BC(name=matching_bcs[1],
                           type='FamilySpecified',
                           point_list=new_bc_pl.reshape((1,-1), order='F'),
                           loc=DIM_TO_LOC[elt_dim],
                           parent=zone_bc_n)
      PT.maia.newDistribution({'Index':new_bc_distrib}, parent=new_bc_n)

      bc_fam_n = PT.get_child_from_label(bc_n, 'FamilyName_t')
      if bc_fam_n is not None:
        PT.new_FamilyName(PT.get_value(bc_fam_n), parent=new_bc_n)
      n_new_elt += bc_pl.size

    # > Update elt node
    ec = np.concatenate([ec]+new_ec)
    PT.set_value(ec_n, ec)
    er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
    er    = PT.get_value(er_n)
    er[1]+= n_new_elt
    PT.set_value(er_n, er)

    # > Update distribution
    elt_distrib_n  = PT.maia.getDistribution(elt_n, distri_name='Element')
    elt_distrib    = PT.get_value(elt_distrib_n)
    elt_distrib[1]+= n_new_elt
    elt_distrib[2]+= n_new_elt
    PT.set_value(elt_distrib_n, elt_distrib)

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
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)==cgns_name

  # > Find BCs described by pls
  bc_nodes = [list(),list()]
  for bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc, depth=2):
    bc_pl = PT.get_child_from_name(bc_n, 'PointList')[1][0]
    for i_side, elt_pl in enumerate([src_pl, tgt_pl]):
      if np.isin(bc_pl, elt_pl, assume_unique=True).all():
        bc_nodes[i_side].append(bc_n)

  # > Get element infos
  matching_bcs = list()
  elt_n = PT.get_child_from_predicate(zone, is_asked_elt)
  if elt_n is not None:
    elt_offset = PT.Element.Range(elt_n)[0]
    elt_size = PT.Element.NVtx(elt_n)
    elt_ec = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
    
    old_to_new_vtx = np.arange(PT.Zone.n_vtx(zone)) + 1
    old_to_new_vtx[src_tgt_vtx[0]-1] = src_tgt_vtx[1] # Normally, elements has no vtx in common
    
    # > Go through BCs described by join vertices and find pairs
    # > Precompute vtx in shared numering only once
    bc_vtx = [list(),list()]
    for i_side, _bc_nodes in enumerate(bc_nodes):
      for src_bc_n in _bc_nodes:
        bc_pl = PT.get_child_from_name(src_bc_n, 'PointList')[1][0]
        ec_idx = np_utils.interweave_arrays([elt_size*(bc_pl-elt_offset)+i_size for i_size in range(elt_size)])
        this_bc_vtx = elt_ec[ec_idx] # List of vertices belonging to bc
        bc_vtx_renum = old_to_new_vtx[this_bc_vtx-1] # Numbering of these vertices in shared numerotation
        bc_vtx[i_side].append(bc_vtx_renum)
    # > Perfom comparaisons
    for src_bc_n, src_bc_vtx in zip(bc_nodes[0], bc_vtx[0]):
      for tgt_bc_n, tgt_bc_vtx in zip(bc_nodes[1], bc_vtx[1]):
        if np.isin(src_bc_vtx, tgt_bc_vtx).all():
          matching_bcs.append([PT.get_name(tgt_bc_n), PT.get_name(src_bc_n)])

  return matching_bcs


def add_undefined_faces(zone, elt_pl, elt_name, vtx_pl, tgt_elt_name):
  '''
  Add faces (TRI_3) in mesh which are face from cells that are not touching the join here described by vtx point_list.
  Check that created face are not already descibed in BCs, or described by two cells (it is an internal face in this case).
  TODO : parallelize this function. Clearer doc
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
  ec_pl  = np_utils.interweave_arrays([elt_size*idx+i_size for i_size in range(elt_size)]) # np_utils.multi_arange(idx*elt_size, (idx+1)*elt_size) seems not to be as quick
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
  tmp_ec   = np.concatenate([tgt_elt_ec, bc_ec])
  tmp_mask = np_utils.is_unique_strided(tmp_ec, size_tgt_elt, method='sort')
  elt_ids = np.where(tmp_mask[0:n_elt_to_add]==True)[0] # Get only tri which are not in BCs
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

  # > Update distribution
  tgt_elt_distrib_n  = PT.maia.getDistribution(tgt_elt_n, distri_name='Element')
  tgt_elt_distrib    = PT.get_value(tgt_elt_distrib_n)
  tgt_elt_distrib[1]+= n_elt_to_add
  tgt_elt_distrib[2]+= n_elt_to_add
  PT.set_value(tgt_elt_distrib_n, tgt_elt_distrib)

  apply_offset_to_elts(zone, n_elt_to_add, tgt_er[1]-n_elt_to_add)

  return new_tgt_elt_pl


def convert_vtx_gcs_as_face_bcs(tree, comm):
  '''
  Convert 1to1, periodic, vertex GCs as FaceCenter BCs for feflo.
  Note : if a face is also present in a BC, then ???
  '''
  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3'
  is_face_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='FaceCenter'
  is_per_gc  = lambda n: PT.get_label(n)=='GridConnectivity_t' and PT.GridConnectivity.is1to1(n) and PT.GridConnectivity.isperiodic(n)

  for zone in PT.get_all_Zone_t(tree):
    # > Get TRI_3 element infos
    elt_n      = PT.get_child_from_predicate(zone, is_tri_elt)
    elt_offset = PT.Element.Range(elt_n)[0]

    # > Use BC PLs to detect BC elts that will be duplicated (cf issue JMarty)
    face_bc_pls = [PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0] 
                      for bc_n in PT.get_nodes_from_predicate(zone, is_face_bc, depth=2)]
    face_bc_pls = np.concatenate(face_bc_pls)-elt_offset+1
        
    zone_bc_n = PT.get_node_from_label(zone, 'ZoneBC_t')
    for gc_n in PT.iter_nodes_from_predicate(zone, is_per_gc, depth=2):
      gc_name = PT.get_name(gc_n)
      gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
      gc_loc  = PT.Subset.GridLocation(gc_n)
      assert gc_loc=='Vertex'

      # > Search faces described by gc vtx
      bc_pl = maia.algo.dist.subset_tools.vtx_ids_to_face_ids(gc_pl, elt_n, comm, True)
      PTP   = EP.PartToPart([face_bc_pls], [bc_pl], comm)
      mask  = np.ones(bc_pl.size, dtype=bool)
      mask[PTP.get_referenced_lnum2()[0]-1] = False
      bc_pl = bc_pl[mask]+elt_offset-1

      if comm.allreduce(bc_pl.size, op=MPI.SUM)!=0:
        bc_n = PT.new_BC(name=gc_name,
                         type='FamilySpecified',
                         point_list=bc_pl.reshape((1,-1), order='F'),
                         loc='FaceCenter',
                         family='GCS',
                         parent=zone_bc_n)
        PT.maia.newDistribution({'Index' : par_utils.dn_to_distribution(bc_pl.size, comm)}, bc_n)


def deplace_periodic_patch(tree, jn_pairs, comm):
  '''
  Use gc_paths and their associated periodic_values in `jn_pairs_and_values` to deplace a range of cell touching
  a GC next to its twin GC (for each pair of GCs), using GCs pl and pld to merge the two domains.
  This function is not working on domains where some vertices are shared by GCs (from different pairs or not).
  '''
  base = PT.get_child_from_label(tree, 'CGNSBase_t')

  zones = PT.get_all_Zone_t(base)
  assert len(zones)==1
  zone = zones[0]

  # > Add GCs as BCs (paraview visu mainly)
  PT.new_Family('PERIODIC', parent=base)
  PT.new_Family('GCS', parent=base)

  n_periodicity = len(jn_pairs)
  new_vtx_nums = list()
  to_constrain_bcs = list()
  matching_bcs   = [dict() for _ in range(n_periodicity)]
  for i_per, gc_paths in enumerate(jn_pairs):

    gc_vtx_n = PT.get_node_from_path(tree, gc_paths[0])
    gc_vtx_pl  = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointList'     ))[0]
    gc_vtx_pld = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointListDonor'))[0]

    # > 1/ Defining the internal surface, that will be constrained in mesh adaptation
    cell_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, 'TETRA_4', MPI.COMM_SELF, elt_full=False) # Tetra made of at least one gc opp vtx
    face_pl = add_undefined_faces(zone, cell_pl, 'TETRA_4', gc_vtx_pld, 'TRI_3') # ?
    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4', MPI.COMM_SELF) # Vertices ids of tetra belonging to cell_pl

    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    new_bc_distrib = par_utils.dn_to_distribution(cell_pl.size, comm)
    bc_n = PT.new_BC(name=cell_bc_name, 
                     type='FamilySpecified',
                     point_list=cell_pl.reshape((1,-1), order='F'),
                     loc='CellCenter',
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':new_bc_distrib}, parent=bc_n)
    
    face_bc_name = f'tri_3_constraint_{i_per}'
    new_bc_distrib = par_utils.dn_to_distribution(face_pl.size, comm)
    bc_n = PT.new_BC(name=face_bc_name, 
                     type='FamilySpecified',
                     point_list=face_pl.reshape((1,-1), order='F'),
                     loc='FaceCenter',
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':new_bc_distrib}, parent=bc_n)
    to_constrain_bcs.append(face_bc_name)

    # maia.io.write_tree(tree, f'OUTPUT/internal_surface_{i_per}.cgns')

    # > 2/ Removing lines defined on join because they surely has their periodic on the other side
    # > Find BCs on GCs that will be deleted because they have their periodic twin
    # > For now only fully described BCs will be treated
    matching_bcs[i_per] = dict()
    bar_to_rm_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, 'BAR_2', MPI.COMM_SELF, elt_full=True) #Bar made of two gc opp vtx
    bar_twins_pl = tag_elmt_owning_vtx(zone, gc_vtx_pl,  'BAR_2', MPI.COMM_SELF, elt_full=True) #Bar made of two gc vtx
    matching_bcs[i_per]['BAR_2'] = find_matching_bcs(zone, bar_to_rm_pl, bar_twins_pl, [gc_vtx_pld, gc_vtx_pl], 'BAR_2')
    bar_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2')
    if bar_n is not None:
      remove_elts_from_pl(zone, bar_n, bar_to_rm_pl, comm)


    # > 3/ Duplicate elts and vtx defined on internal created surface, and updating vtx numbering
    #      of elmts touching this surface
    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_pl
    to_update_face_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'TRI_3', MPI.COMM_SELF, elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'BAR_2', MPI.COMM_SELF, elt_full=True)

    # > Ambiguous faces that contains all vtx but are not included in patch cells can be removed
    to_update_face_pl = is_elt_included(zone, to_update_face_pl, 'TRI_3', cell_pl, 'TETRA_4')

    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}
  
    face_bc_name = f'tri_3_periodic_{i_per}'
    new_vtx_num = duplicate_elts(zone, face_pl, 'TRI_3', f'tri_3_periodic_{i_per}', elts_to_update, comm)
    to_constrain_bcs.append(face_bc_name)


    # > 4/ Apply periodic transformation to vtx and flowsol
    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4', MPI.COMM_SELF)
    perio_val = PT.GridConnectivity.periodic_values(PT.get_node_from_path(tree, gc_paths[1]))
    periodic = perio_val.asdict(snake_case=True)

    dist_transform.transform_affine_zone(zone, vtx_pl, MPI.COMM_SELF, **periodic, apply_to_fields=True)

    # maia.io.write_tree(tree, f'OUTPUT/deplaced_{i_per}.cgns')


    # > 5/ Merge two GCs that are now overlaping
    n_vtx = PT.Zone.n_vtx(zone)
    bc_name1 = PT.path_tail(gc_paths[0])
    bc_name2 = PT.path_tail(gc_paths[1])
    vtx_match_num = [gc_vtx_pl, gc_vtx_pld]
    vtx_tag = np.arange(1, n_vtx+1, dtype=np.int32)
    old_to_new_vtx = merge_periodic_bc(zone, (bc_name1, bc_name2), vtx_tag, vtx_match_num, comm, keep_original=True)
    
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


def retrieve_initial_domain(tree, jn_pairs_and_values, new_vtx_num, bcs_to_retrieve, comm):
  '''
  Use `gc_paths` and their associated `periodic_values` to deplace the adapted range of cells that have been adapted to its initial position,
  using `new_vtx_num` match info to merge the two domains.
  Twin BCs defined in GCs that have been deleted while using `deplace_periodic_patch` can be retrieved thanks to `bcs_to_retrieve`.
  '''
  n_periodicity = len(jn_pairs_and_values.keys())

  base = PT.get_child_from_label(tree, 'CGNSBase_t')
  is_3d = PT.get_value(base)[0]==3

  zone = PT.get_child_from_label(base, 'Zone_t')
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  cell_elt_name = 'TETRA_4' if is_3d else 'TRI_3'
  face_elt_name = 'TRI_3'   if is_3d else 'BAR_2'
  edge_elt_name = 'BAR_2'   if is_3d else  None

  # > Removing old periodic patch
  i_per = n_periodicity-1
  for gc_paths, periodic_values in reversed(jn_pairs_and_values.items()): # reversed for future multiperiodic
  
    # > 1/ Get elts and vtx for BCs out of feflo
    #      TODO: some TRI_3 should be avoided with is_elt_included again
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    cell_bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_bc_pl = PT.get_value(PT.Subset.getPatch(cell_bc_n))[0]
    vtx_pl = elmt_pl_to_vtx_pl(zone, cell_bc_pl, 'TETRA_4', MPI.COMM_SELF)

    still_here_gc_name  = gc_paths[0].split('/')[-1]
    to_retrieve_gc_name = gc_paths[1].split('/')[-1]
    bc_n = PT.get_child_from_name(zone_bc_n, still_here_gc_name)
    face_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]

    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_bc_pl
    to_update_face_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'TRI_3', MPI.COMM_SELF, elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(zone, vtx_pl, 'BAR_2', MPI.COMM_SELF, elt_full=True)
    to_update_face_pl = is_elt_included(zone, to_update_face_pl, 'TRI_3', cell_bc_pl, 'TETRA_4')
    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}

    # > 2/ Duplicate GC surface and update element connectivities in the patch
    duplicate_elts(zone, face_pl, 'TRI_3', to_retrieve_gc_name, elts_to_update, comm, elt_duplicate_bcs=bcs_to_retrieve[i_per])

    # > 3/ Deplace periodic patch to retrieve initial domain
    #      (vtx_pl is updated because has changed with surface duplication)
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
    vtx_pl  = elmt_pl_to_vtx_pl(zone, cell_pl, 'TETRA_4', MPI.COMM_SELF)
    periodic = periodic_values[0].asdict(True)
    dist_transform.transform_affine_zone(zone, vtx_pl, MPI.COMM_SELF, **periodic, apply_to_fields=True)
    # maia.io.write_tree(tree, f'OUTPUT/adapted_and_deplaced_{i_per}.cgns')

    # > 4-5/ Merge two constraint surfaces
    vtx_tag_n = PT.get_node_from_name(zone, 'vtx_tag')
    vtx_tag   = PT.get_value(vtx_tag_n)
    _ = merge_periodic_bc(zone,
                      [f'{face_elt_name.lower()}_constraint_{i_per}', f'{face_elt_name.lower()}_periodic_{i_per}'],
                      vtx_tag,
                      new_vtx_num[i_per], comm)
    i_per -=1

  rm_feflo_added_elt(zone, comm)


def rm_feflo_added_elt(zone, comm):
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
      is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                               PT.Element.CGNSName(n)==LOC_TO_CGNS[bc_loc]
      elt_n = PT.get_child_from_predicate(zone, is_asked_elt)
      if elt_n is not None:
        remove_elts_from_pl(zone, elt_n, bc_pl, comm)
