import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
import maia.transfer.protocols as EP
from   maia.utils  import np_utils, par_utils, as_pdm_gnum
from   maia.utils.parallel import algo as par_algo
from   maia.algo.dist import transform as dist_transform
from   maia.algo.dist.merge_ids      import merge_distributed_ids
from   maia.algo.dist.remove_element import remove_elts_from_pl
from   maia.algo.dist.subset_tools   import vtx_ids_to_face_ids

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

def duplicate_specified_vtx(zone, vtx_pl, comm):
  """
  Duplicate vtx specified in `vtx_pl` from a distributed zone. Vertex base FlowSolution_t
  are duplicated as well.
  Size of zone and Vertex distribution are updated
  
  Note : if an id appear twice in vtx_pl, it will be duplicated twice. This is because
  data are added to the rank requesting the vertices in vtx_pl
  """
  distri_n = PT.maia.getDistribution(zone, 'Vertex')
  distri   = PT.get_value(distri_n)
  dn_vtx   = distri[1] - distri[0] # Initial number of vertices
  n_vtx    = distri[2]

  # Update GridCoordinates
  coords_keys = ['CoordinateX', 'CoordinateY', 'CoordinateZ']
  coord_nodes = {key: PT.get_node_from_name(zone, key) for key in coords_keys}
  coords = {key: PT.get_value(node) for key, node in coord_nodes.items()}
  new_coords = EP.block_to_part(coords, distri, [vtx_pl], comm)
  for key in coords_keys:
    PT.set_value(coord_nodes[key], np.concatenate([coords[key], new_coords[key][0]]))

  # Update FlowSolution
  is_loc_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_loc_fs):
    assert PT.get_child_from_name(fs_n, 'PointList') is None, "Partial FS are not supported"

    arrays_n = PT.get_children_from_label(fs_n, 'DataArray_t')
    arrays = {PT.get_name(array_n) : PT.get_value(array_n) for array_n in arrays_n}
    new_arrays = EP.block_to_part(arrays, distri, [vtx_pl], comm)
    for array_n in arrays_n:
      key = PT.get_name(array_n)
      PT.set_value(array_n, np.concatenate([arrays[key], new_arrays[key][0]]))

  # Update distribution and zone size
  PT.get_value(zone)[0][0] += comm.allreduce(vtx_pl.size, op=MPI.SUM)
  add_distri = par_utils.dn_to_distribution(vtx_pl.size, comm)
  new_distri = distri+add_distri
  PT.set_value(distri_n, new_distri)

  # > Replace added vertices at the end of array
  old_gnum = np.arange(          distri[0],          distri[1])+1
  new_gnum = np.arange(n_vtx+add_distri[0],n_vtx+add_distri[1])+1
  vtx_gnum = np.concatenate([old_gnum, new_gnum])

  # Update GridCoordinates
  coords_keys = ['CoordinateX', 'CoordinateY', 'CoordinateZ']
  coord_nodes = {key: PT.get_node_from_name(zone, key) for key in coords_keys}
  coords = {key: [PT.get_value(node)] for key, node in coord_nodes.items()}
  new_coords = EP.part_to_block(coords, new_distri, [vtx_gnum], comm)
  for key in coords_keys:
    PT.set_value(coord_nodes[key], new_coords[key])

  # Update FlowSolution
  is_loc_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_loc_fs):
    assert PT.get_child_from_name(fs_n, 'PointList') is None, "Partial FS are not supported"

    arrays_n = PT.get_children_from_label(fs_n, 'DataArray_t')
    arrays = {PT.get_name(array_n) : [PT.get_value(array_n)] for array_n in arrays_n}
    new_arrays = EP.part_to_block(arrays, new_distri, [vtx_gnum], comm)
    for array_n in arrays_n:
      key = PT.get_name(array_n)
      PT.set_value(array_n, new_arrays[key])

def remove_specified_vtx(zone, vtx_pl, comm):
  """
  Remove vtx specified in `vtx_pl` from a distributed zone. Vertex base FlowSolution_t
  are removed as well.
  Size of zone and Vertex distribution are updated
  
  Note : id can appear twice in vtx_pl, it will be removed only once
  """
  distri_n = PT.maia.getDistribution(zone, 'Vertex')
  distri   = PT.get_value(distri_n)
  dn_vtx   = distri[1] - distri[0] # Initial number of vertices
  
  # Get ids to remove
  ptb = EP.PartToBlock(distri, [vtx_pl], comm)
  ids = ptb.getBlockGnumCopy()-distri[0]-1

  # Update GridCoordinates
  for grid_co_n in PT.get_children_from_predicate(zone, 'GridCoordinates_t'):
    for da_n in PT.get_children_from_label(grid_co_n, 'DataArray_t'):
      old_val = PT.get_value(da_n)
      PT.set_value(da_n, np.delete(old_val, ids))

  # Update FlowSolution
  is_loc_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_loc_fs):
    assert PT.get_child_from_name(fs_n, 'PointList') is None, "Partial FS are not supported"
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      old_val = PT.get_value(da_n)
      PT.set_value(da_n, np.delete(old_val, ids))

  # Update distribution and zone size
  PT.get_value(zone)[0][0] -= comm.allreduce(ids.size, op=MPI.SUM)
  PT.set_value(distri_n, par_utils.dn_to_distribution(dn_vtx - ids.size, comm))


def elmt_pl_to_vtx_pl(zone, elt_n, elt_pl, comm):
  '''
  Return distributed gnum of vertices describing elements tagged in `elt_pl`.
  '''
  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]

  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  elt_distri = PT.maia.getDistribution(elt_n, 'Element')[1]

  # > Get partitionned connectivity of elt_pl
  elt_ec   = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
  ids      = elt_pl - elt_offset +1
  _, pl_ec = EP.block_to_part_strided(elt_size, elt_ec, elt_distri, [ids], comm)

  # > Get distributed vertices gnum referenced in pl_ec 
  ptb    = EP.PartToBlock(vtx_distri, pl_ec, comm)
  vtx_pl = ptb.getBlockGnumCopy()

  return vtx_pl


def tag_elmt_owning_vtx(elt_n, vtx_pl, comm, elt_full=False):
  '''
  Return the point_list of elements that owns one or all of their vertices in the vertex point_list.
  Important : elt_pl is returned as a distributed array, w/o any assumption on the holding
  rank : vertices given by a rank can spawn an elt_idx on a other rank.
  '''
  if elt_n is not None:
    elt_offset = PT.Element.Range(elt_n)[0]
    gc_elt_pl  = vtx_ids_to_face_ids(vtx_pl, elt_n, comm, elt_full)+elt_offset-1
  else:
    gc_elt_pl = np.empty(0, dtype=vtx_pl.dtype)
  return gc_elt_pl

def find_shared_faces(tri_elt, tri_pl, tetra_elt, tetra_pl, comm):
  """
  For the given TRI and TETRA elements node, search the TRI faces shared
  by the TETRA elements. In addition, TRI and TETRA can be filtered
  with tri_pl and tetra_pl, which are distributed - cgnslike list of indices.
  """
  # TRI elts
  #   Get ec
  src_distri    = PT.maia.getDistribution(tri_elt, 'Element')[1]
  size_src_elt  = PT.Element.NVtx(tri_elt)
  src_ec        = PT.get_child_from_name(tri_elt, 'ElementConnectivity')[1]
  #   Get list of TRI faces to select from other ranks
  ptb = EP.PartToBlock(src_distri, [tri_pl - PT.Element.Range(tri_elt)[0] + 1], comm)
  src_dist_gnum= ptb.getBlockGnumCopy() + PT.Element.Range(tri_elt)[0] - 1
  src_dist_ids = ptb.getBlockGnumCopy() - src_distri[0] - 1
  #   Extract connectivity
  src_ec_idx  = np_utils.interweave_arrays([size_src_elt*src_dist_ids+i_size for i_size in range(size_src_elt)])
  src_ec_elt = src_ec[src_ec_idx]

  # TETRA elts
  #   Get ec
  tgt_distri   = PT.maia.getDistribution(tetra_elt, 'Element')[1]
  size_tgt_elt = PT.Element.NVtx(tetra_elt)
  tgt_ec       = PT.get_child_from_name(tetra_elt, 'ElementConnectivity')[1]
  #   Get list of TETRA elts to select from other ranks
  ptb = EP.PartToBlock(tgt_distri, [tetra_pl - PT.Element.Range(tetra_elt)[0] + 1], comm)
  tgt_dist_ids = ptb.getBlockGnumCopy() - tgt_distri[0] - 1
  #   Extract connectivity
  tgt_ec_idx  = np_utils.interweave_arrays([size_tgt_elt*tgt_dist_ids+i_size for i_size in range(size_tgt_elt)])
  tgt_ec_elt = tgt_ec[tgt_ec_idx]
  

  # Tri are already decomposed
  src_face_vtx_idx, src_face_vtx = 3*np.arange(src_dist_ids.size+1, dtype=np.int32), src_ec_elt
  # Do tetra decomposition, locally
  tgt_face_vtx_idx, tgt_face_vtx = PDM.decompose_std_elmt_faces(PDM._PDM_MESH_NODAL_TETRA4, as_pdm_gnum(tgt_ec_elt))

  # Now we we will reuse seq algorithm is_unique_strided to find
  # shared faces. Since they can be on different processes, we need to gather it
  # (according to their sum of vtx) before.
  # A fully distributded version of is_unique_strided would be better ...
  tri_key   = np.add.reduceat(src_face_vtx, src_face_vtx_idx[:-1])
  tetra_key = np.add.reduceat(tgt_face_vtx, tgt_face_vtx_idx[:-1])

  weights = [np.ones(t.size, float) for t in [tri_key, tetra_key]]
  ptb = EP.PartToBlock(None, [tri_key, tetra_key], comm, weight=weights, keep_multiple=True)
  cst_stride = [np.ones(t.size-1, np.int32) for t in [src_face_vtx_idx, tgt_face_vtx_idx]]

  # Origin is not mandatory for TETRA because we just want the TRI ids at the end
  _, origin = ptb.exchange_field([src_dist_gnum, np.zeros(tetra_key.size, src_dist_gnum.dtype)], part_stride=cst_stride)
  _, tmp_ec = ptb.exchange_field([src_face_vtx, tgt_face_vtx], part_stride=[3*s for s in cst_stride])
  mask = np_utils.is_unique_strided(tmp_ec, 3, method='hash')

  mask[origin == 0] = True # We dont want to get tetra faces
  out = origin[~mask]

  return out


def update_elt_vtx_numbering(zone, elt_n, old_to_new_vtx, comm, elt_pl=None):
  '''
  Update element connectivity (partialy if `elt_pl` provided) according to the new vertices numbering described in `old_to_new_vtx`.
  '''
  if elt_n is not None:
    ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
    ec    = PT.get_value(ec_n)

    if elt_pl is None:
      vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
      ec = EP.block_to_part(old_to_new_vtx, vtx_distri, [ec], comm)[0]
    else:
      elt_size   = PT.Element.NVtx(elt_n)
      elt_offset = PT.Element.Range(elt_n)[0]
      elt_distri = PT.maia.getDistribution(elt_n, 'Element')[1]
      vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]

      elt_pl_shft = elt_pl - elt_offset +1
      ptb = EP.PartToBlock(elt_distri, [elt_pl_shft], comm)
      ids = ptb.getBlockGnumCopy()-elt_distri[0]-1
      ec_ids = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
      new_num_ec = EP.block_to_part(old_to_new_vtx, vtx_distri, [ec[ec_ids]], comm)[0]
      ec[ec_ids] = new_num_ec

    PT.set_value(ec_n, ec)


def apply_offset_to_elts(zone, offset, min_range):
  '''
  Go through all elements with ElementRange>min_range, applying offset to their ElementRange.
  '''
  # > Add offset to elements with ElementRange>min_range
  elt_nodes = PT.Zone.get_ordered_elements(zone)
  for elt_n in elt_nodes:
    elt_range = PT.Element.Range(elt_n)
    if elt_range[0] > min_range:
      elt_range += offset

  # > Treating all BCs outside of elts because if not elt of dim of BC,
  #   it wont be treated.
  for bc_n in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
    pl = PT.get_child_from_name(bc_n, 'PointList')[1][0]
    pl[min_range<pl] += offset

def merge_periodic_bc(zone, bc_names, vtx_tag, old_to_new_vtx_num, comm, keep_original=False):
  '''
  Merge two similar BCs using a vtx numbering and a table describing how to transform vertices from first BC to second BC vertices.
  First BC can be kept using `keep_original` argument.
  '''
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  vtx_distri   = PT.maia.getDistribution(zone, 'Vertex')[1]

  # TODO: directement choper les GCs
  pbc1_n      = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_loc    = PT.Subset.GridLocation(pbc1_n)
  pbc1_pl     = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  elt_n       = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)==LOC_TO_CGNS[pbc1_loc])
  pbc1_vtx_pl = elmt_pl_to_vtx_pl(zone, elt_n, pbc1_pl, comm)

  pbc2_n      = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_loc    = PT.Subset.GridLocation(pbc2_n)
  pbc2_pl     = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  elt_n       = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)==LOC_TO_CGNS[pbc2_loc])
  pbc2_vtx_pl = elmt_pl_to_vtx_pl(zone, elt_n, pbc2_pl, comm)

  old_vtx_num = old_to_new_vtx_num[0]
  new_vtx_num = old_to_new_vtx_num[1]

  ptb = EP.PartToBlock(vtx_distri, [pbc1_vtx_pl], comm)
  pbc1_vtx_pl  = ptb.getBlockGnumCopy()
  pbc1_vtx_ids = pbc1_vtx_pl-vtx_distri[0]
  pl1_tag = vtx_tag[pbc1_vtx_ids-1]

  ptb = EP.PartToBlock(vtx_distri, [pbc2_vtx_pl], comm)
  pbc2_vtx_pl  = ptb.getBlockGnumCopy()
  pbc2_vtx_ids = pbc2_vtx_pl-vtx_distri[0]
  pl2_tag = vtx_tag[pbc2_vtx_ids-1]

  ptp = EP.PartToPart([pl1_tag], [old_vtx_num], comm)
  request1 = ptp.iexch( PDM._PDM_MPI_COMM_KIND_P2P,
                        PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2, 
                        [pbc1_vtx_pl])
  _, gnum1_vtx = ptp.wait(request1)
  gnum1_vtx = gnum1_vtx[0]

  _part1_to_part2_idx = np.arange(old_vtx_num.size+1, dtype=np.int32)
  ptp = PDM.PartToPart(comm, [as_pdm_gnum(old_vtx_num)], [as_pdm_gnum(pl2_tag)], [_part1_to_part2_idx], [as_pdm_gnum(new_vtx_num)])
  request1 = ptp.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                       PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2, 
                       [gnum1_vtx])
  _, part_data = ptp.wait(request1)

  old_to_new_vtx = merge_distributed_ids(vtx_distri, pbc2_vtx_pl, part_data[0], comm, False)

  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==LOC_TO_CGNS[pbc2_loc]
  elt_n = PT.get_child_from_predicate(zone, is_asked_elt)
  if not keep_original:
    remove_elts_from_pl(zone, elt_n, pbc1_pl, comm)
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  remove_elts_from_pl(zone, elt_n, pbc2_pl, comm)

  tet_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4')
  tri_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3')
  bar_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2')

  update_elt_vtx_numbering(zone, tet_n, old_to_new_vtx, comm)
  update_elt_vtx_numbering(zone, tri_n, old_to_new_vtx, comm)
  update_elt_vtx_numbering(zone, bar_n, old_to_new_vtx, comm)

  # > Update Vertex BCs and GCs
  update_vtx_bnds(zone, old_to_new_vtx, comm)

  # > Remove coordinates and FS + udpate zone dim
  remove_specified_vtx(zone, pbc2_vtx_pl, comm)

  return old_to_new_vtx


def update_vtx_bnds(zone, old_to_new_vtx, comm):
  '''
  Update Vertex BCs and GCs according to the new vertices numbering described in `old_to_new_vtx`.
  TODO: predicates
  '''
  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  if zone_bc_n is not None:
    is_vtx_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)=='Vertex'
    for bc_n in PT.get_children_from_predicate(zone_bc_n, is_vtx_bc):
      bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
      bc_pl   = PT.get_value(bc_pl_n)[0]
      bc_pl   = EP.block_to_part(old_to_new_vtx, vtx_distri, [bc_pl], comm)[0]
      assert (bc_pl!=-1).all()
      PT.set_value(bc_pl_n, bc_pl.reshape((1,-1), order='F'))

  zone_gc_n = PT.get_child_from_label(zone, 'ZoneGridConnectivity_t')
  if zone_gc_n is not None:
    is_vtx_gc = lambda n: PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t']  and\
                          PT.Subset.GridLocation(n)=='Vertex'
    for gc_n in PT.get_children_from_predicate(zone_gc_n, is_vtx_gc):
      gc_pl_n = PT.get_child_from_name(gc_n, 'PointList')
      gc_pl   = PT.get_value(gc_pl_n)[0]
      gc_pl   = EP.block_to_part(old_to_new_vtx, vtx_distri, [gc_pl], comm)[0]
      assert (gc_pl!=-1).all()
      PT.set_value(gc_pl_n, gc_pl.reshape((1,-1), order='F'))

      gc_pld_n = PT.get_child_from_name(gc_n, 'PointListDonor')
      gc_pld   = PT.get_value(gc_pld_n)[0]
      gc_pld   = EP.block_to_part(old_to_new_vtx, vtx_distri, [gc_pld], comm)[0]
      assert (gc_pld!=-1).all()
      PT.set_value(gc_pld_n, gc_pld.reshape((1,-1), order='F'))


def duplicate_elts(zone, elt_n, elt_pl, as_bc, elts_to_update, comm, elt_duplicate_bcs=[]):
  '''
  Duplicate elements tagged in `elt_pl` by updating its ElementConnectivity and ElementRange nodes,
  as well as ElementRange nodes of elements with inferior dimension (assuming that element nodes are organized with decreasing dimension order).
  Created elements can be tagged in a new BC_t through `as_bc` argument.
  Elements of other dimension touching vertices of duplicated elements can be updating with new created vertices through `elts_to_update` argument.
  BCs fully described by duplicated element vertices can be duplicated as well using `elt_duplicate_bcs` argument (lineic BCs in general).
  '''
  # > Add duplicated vertex
  n_vtx = PT.Zone.n_vtx(zone)
  elt_vtx_pl   = elmt_pl_to_vtx_pl(zone, elt_n, elt_pl, comm)
  n_vtx_to_add = elt_vtx_pl.size

  duplicate_specified_vtx(zone, elt_vtx_pl, comm)
  
  new_vtx_distri = par_utils.dn_to_distribution(n_vtx_to_add, comm)
  new_vtx_pl     = np.arange(n_vtx+new_vtx_distri[0],n_vtx+new_vtx_distri[1], dtype=elt_vtx_pl.dtype)+1
  vtx_distri     = PT.maia.getDistribution(zone, 'Vertex')[1]
  new_vtx_num    = [elt_vtx_pl,new_vtx_pl]

  new_vtx_pl = EP.part_to_block([new_vtx_pl], vtx_distri, [elt_vtx_pl], comm)
  ptb = EP.PartToBlock(vtx_distri, [elt_vtx_pl], comm)
  elt_vtx_ids = ptb.getBlockGnumCopy()-vtx_distri[0]-1

  old_to_new_vtx = np.arange(vtx_distri[0],vtx_distri[1], dtype=vtx_distri.dtype)+1
  old_to_new_vtx[elt_vtx_ids] = new_vtx_pl
  
  # > Add duplicated elements
  n_elt      = PT.Element.Size(elt_n)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  elt_dim    = PT.Element.Dimension(elt_n)
  elt_distri_n = PT.maia.getDistribution(elt_n, distri_name='Element')
  elt_distri   = PT.get_value(elt_distri_n)
  
  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)

  # > Copy elements connectivity 
  elt_pl_shft = elt_pl - elt_offset +1
  ptb = EP.PartToBlock(elt_distri, [elt_pl_shft], comm)
  ids = ptb.getBlockGnumCopy()-elt_distri[0]-1
  n_elt_to_add = ids.size
  ec_ids = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
  duplicated_ec = EP.block_to_part(old_to_new_vtx, vtx_distri, [ec[ec_ids]], comm)[0]
  new_ec = np.concatenate([ec, duplicated_ec])
  
  # > Update element distribution
  add_elt_distri = par_utils.dn_to_distribution(n_elt_to_add, comm)
  new_elt_distri = elt_distri+add_elt_distri
  PT.set_value(elt_distri_n, new_elt_distri)

  # > Update ElementConnectivity, by adding new elements at the end of distribution
  n_elt = elt_distri[2]
  old_gnum = np.arange(          elt_distri[0],          elt_distri[1])+1
  new_gnum = np.arange(n_elt+add_elt_distri[0],n_elt+add_elt_distri[1])+1
  elt_gnum = np.concatenate([old_gnum, new_gnum])

  cst_stride = np.full(elt_gnum.size, elt_size, np.int32)
  ptb = EP.PartToBlock(new_elt_distri, [elt_gnum], comm)
  _, new_ec = ptb.exchange_field([new_ec], part_stride=[cst_stride])

  PT.set_value(ec_n, new_ec)

  # > Update ElementRange
  er   = PT.Element.Range(elt_n)
  er[1] += add_elt_distri[2]

  apply_offset_to_elts(zone, add_elt_distri[2], er[1]-add_elt_distri[2])

  # > Create associated BC if asked
  if as_bc is not None:
    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    new_elt_pl = np.arange(n_elt+add_elt_distri[0],
                           n_elt+add_elt_distri[1], dtype=elt_distri.dtype) + elt_offset
    bc_n = PT.new_BC(name=as_bc, 
                     type='FamilySpecified',
                     point_list=new_elt_pl.reshape((1,-1), order='F'),
                     loc=DIM_TO_LOC[elt_dim],
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':add_elt_distri}, parent=bc_n)

  n_tri_added_tot = add_elt_distri[2]

  # > Duplicate twin BCs
  new_ec = list()
  twin_elt_bc_pl = list()

  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2'
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  if elt_duplicate_bcs != []:
    assert elt_n is not None
    n_elt        = PT.Element.Size(elt_n)
    elt_size     = PT.Element.NVtx(elt_n)
    elt_offset   = PT.Element.Range(elt_n)[0]
    elt_dim      = PT.Element.Dimension(elt_n)
    elt_distri_n = PT.maia.getDistribution(elt_n, 'Element')
    elt_distri   = PT.get_value(elt_distri_n)

    ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
    ec    = PT.get_value(ec_n)

    n_new_elt = 0
    add_elt_distri = np.zeros(3, dtype=elt_distri.dtype)
    new_elt_distri = elt_distri
    for matching_bcs in elt_duplicate_bcs:
      bc_n = PT.get_child_from_name(zone_bc_n, matching_bcs[0])
      bc_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
      twin_elt_bc_pl.append(bc_pl)

      bc_pl_shft = bc_pl - elt_offset +1
      ptb = EP.PartToBlock(new_elt_distri, [bc_pl_shft], comm)
      ids = ptb.getBlockGnumCopy()-new_elt_distri[0]-1
      n_elt_to_add_l = ids.size
      ec_ids = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
      new_bc_ec = EP.block_to_part(old_to_new_vtx, vtx_distri, [ec[ec_ids]], comm)[0]
      new_ec.append(new_bc_ec)

      # > Compute element distribution
      add_elt_distri_l = par_utils.dn_to_distribution(n_elt_to_add_l, comm)
      add_elt_distri  += add_elt_distri_l
      new_elt_distri   = new_elt_distri+add_elt_distri_l

      new_bc_pl = np.arange(n_elt+add_elt_distri_l[0],
                            n_elt+add_elt_distri_l[1], dtype=bc_pl.dtype) + elt_offset
      
      new_bc_n = PT.new_BC(name=matching_bcs[1],
                           type='FamilySpecified',
                           point_list=new_bc_pl.reshape((1,-1), order='F'),
                           loc=DIM_TO_LOC[elt_dim],
                           parent=zone_bc_n)
      PT.maia.newDistribution({'Index':add_elt_distri_l}, parent=new_bc_n)

      bc_fam_n = PT.get_child_from_label(bc_n, 'FamilyName_t')
      if bc_fam_n is not None:
        PT.new_FamilyName(PT.get_value(bc_fam_n), parent=new_bc_n)
      n_new_elt += bc_pl.size

    # > Update ElementConnectivity, by adding new elements at the end of distribution
    new_ec = np.concatenate([ec]+new_ec)

    n_elt = elt_distri[2]
    old_gnum = np.arange(          elt_distri[0],          elt_distri[1])+1
    new_gnum = np.arange(n_elt+add_elt_distri[0],n_elt+add_elt_distri[1])+1
    elt_gnum = np.concatenate([old_gnum, new_gnum])

    cst_stride = np.full(elt_gnum.size, elt_size, np.int32)
    ptb = EP.PartToBlock(new_elt_distri, [elt_gnum], comm)
    _, new_ec = ptb.exchange_field([new_ec], part_stride=[cst_stride])

    PT.set_value(ec_n, new_ec)

    # > Update ElementRange
    er   = PT.Element.Range(elt_n)
    er[1] += add_elt_distri[2]

    # > Update distribution
    PT.set_value(elt_distri_n, new_elt_distri)

  # > Update vtx numbering of elements in patch to separate patch
  cell_pl = elts_to_update['TETRA_4']
  face_pl = elts_to_update['TRI_3']
  line_pl = elts_to_update['BAR_2']+n_tri_added_tot # Attention au décalage de la PL 

  # > Exclude constraint surf or both will move
  tag_face = par_algo.gnum_isin(face_pl, elt_pl, comm)
  face_pl  = face_pl[~tag_face]
  bc_line_pl = np.concatenate(twin_elt_bc_pl) if len(twin_elt_bc_pl)!=0 else np.empty(0, dtype=int)
  tag_line = par_algo.gnum_isin(line_pl, bc_line_pl, comm) 
  line_pl  = line_pl[~tag_line]

  tet_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4')
  tri_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3')
  bar_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2')

  update_elt_vtx_numbering(zone, tet_n, old_to_new_vtx, comm, elt_pl=cell_pl)
  update_elt_vtx_numbering(zone, tri_n, old_to_new_vtx, comm, elt_pl=face_pl)
  update_elt_vtx_numbering(zone, bar_n, old_to_new_vtx, comm, elt_pl=line_pl)

  return new_vtx_num

    
def find_matching_bcs(zone, elt_n, src_pl, tgt_pl, src_tgt_vtx, comm):
  '''
  Find pairs of twin BCs (lineic in general) using a matching vtx table.
  '''
  # Steps:
  #  1. Create a old->new table for vtx, using gc information
  #  2. Pre filter BCs : keep the one appearing 100% in src_pl or tgt_pl.
  #  3. For these bcs : find vertices ids in BAR, and requested vtx new id with old->new
  #  4. Then, compare all arrays to find src / tgt matches

  matching_bcs = list()

  elt_dim = PT.Element.Dimension(elt_n)
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)==DIM_TO_LOC[elt_dim]

  # > Compute new vtx numbering merging vtx from `src_tgt_vtx`
  #   Maybe there will be an issue in axisym because of vtx in both GCs
  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
  old_to_new_vtx = merge_distributed_ids(vtx_distri, src_tgt_vtx[0], src_tgt_vtx[1], comm, False)

  # > Find BCs described by element pls
  bc_nodes = [list(),list()]
  for bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc, depth=2):
    bc_pl = PT.get_child_from_name(bc_n, 'PointList')[1][0]
    for i_side, elt_pl in enumerate([src_pl, tgt_pl]):
      mask = par_algo.gnum_isin(bc_pl, elt_pl, comm)
      if par_utils.all_true([mask], lambda t:t.all(), comm):
        bc_nodes[i_side].append(bc_n)

  # > Get element infos
  elt_offset = PT.Element.Range(elt_n)[0]
  elt_size   = PT.Element.NVtx(elt_n)
  elt_ec     = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
  elt_distri = PT.maia.getDistribution(elt_n, 'Element')[1]
  
  # > Precompute vtx in shared numbering
  bc_vtx = [list(),list()]
  for i_side, _bc_nodes in enumerate(bc_nodes):
    for src_bc_n in _bc_nodes:
      bc_pl = PT.get_child_from_name(src_bc_n, 'PointList')[1][0]
      
      # > Get BC connectivity
      ptb = EP.PartToBlock(elt_distri, [bc_pl-elt_offset+1], comm)
      ids = ptb.getBlockGnumCopy()-elt_distri[0]-1
      ec_idx = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)])
      this_bc_vtx = elt_ec[ec_idx] # List of vertices belonging to bc

      # > Set BC connectivity in vtx shared numbering
      vtx_ptb = EP.PartToBlock(vtx_distri, [this_bc_vtx], comm)
      vtx_ids = vtx_ptb.getBlockGnumCopy()
      bc_vtx_renum = EP.block_to_part(old_to_new_vtx, vtx_distri, [vtx_ids], comm) # Numbering of these vertices in shared numerotation
      
      bc_vtx[i_side].append(bc_vtx_renum[0])

  # > Perfom comparaisons
  for src_bc_n, src_bc_vtx in zip(bc_nodes[0], bc_vtx[0]):
    for tgt_bc_n, tgt_bc_vtx in zip(bc_nodes[1], bc_vtx[1]):
      tgt_vtx_tag = par_algo.gnum_isin(tgt_bc_vtx, src_bc_vtx, comm)
      if par_utils.all_true([tgt_vtx_tag], lambda t:t.all(), comm):
        matching_bcs.append([PT.get_name(tgt_bc_n), PT.get_name(src_bc_n)])

  return matching_bcs


def add_undefined_faces(zone, elt_n, elt_pl, tgt_elt_n, comm):
  '''
  Decompose `elt_pl` tetra faces (which are triangles), adding those that are not already 
  defined in zone and not defined by two tetras.
  '''
  # > Get element infos
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  ec_n       = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec         = PT.get_value(ec_n)
  elt_name   = PT.Element.CGNSName(elt_n)
  elt_distri = PT.maia.getDistribution(elt_n, 'Element')[1]
  assert elt_name=='TETRA_4'

  tgt_elt_size   = PT.Element.NVtx(tgt_elt_n)
  tgt_elt_offset = PT.Element.Range(tgt_elt_n)[0]
  tgt_ec_n       = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec         = PT.get_value(tgt_ec_n)
  tgt_elt_name   = PT.Element.CGNSName(tgt_elt_n)
  tgt_elt_distri_n = PT.maia.getDistribution(tgt_elt_n, distri_name='Element')
  tgt_elt_distri   = PT.get_value(tgt_elt_distri_n)
  assert tgt_elt_name=='TRI_3'

  # > Get TETRA_4 elt_pl connectivity
  elt_pl_shft = elt_pl - elt_offset +1
  ptb = EP.PartToBlock(elt_distri, [elt_pl_shft], comm)
  ids = ptb.getBlockGnumCopy()-elt_distri[0]-1
  ec_pl  = np_utils.interweave_arrays([elt_size*ids+i_size for i_size in range(elt_size)]) # np_utils.multi_arange(idx*elt_size, (idx+1)*elt_size) seems not to be as quick
  ec_elt = ec[ec_pl]

  # > Decompose tetra faces 
  tgt_face_vtx_idx, tgt_elt_ec = PDM.decompose_std_elmt_faces(PDM._PDM_MESH_NODAL_TETRA4, as_pdm_gnum(ec_elt))
  n_elt_to_add = tgt_face_vtx_idx.size-1

  # > Find faces not already defined in TRI_3 connectivity or duplicated
  tmp_ec  = np.concatenate([tgt_elt_ec, tgt_ec])
  l_mask  = par_algo.is_unique_strided(tmp_ec, tgt_elt_size, comm)
  elt_ids = np.where(l_mask[0:n_elt_to_add])[0]

  n_elt_to_add = elt_ids.size
  ec_pl = np_utils.interweave_arrays([tgt_elt_size*elt_ids+i_size for i_size in range(tgt_elt_size)])
  tgt_elt_ec = tgt_elt_ec[ec_pl]

  # > Update distribution
  add_elt_distri = par_utils.dn_to_distribution(n_elt_to_add, comm)
  new_elt_distri = tgt_elt_distri+add_elt_distri
  PT.set_value(tgt_elt_distri_n, new_elt_distri)

  # > Update target element
  tgt_ec_n   = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec     = PT.get_value(tgt_ec_n)
  tgt_new_ec = np.concatenate([tgt_ec, tgt_elt_ec])

  # > Replace added elements at the end of distribution
  tgt_n_elt = tgt_elt_distri[2]
  old_gnum = np.arange(          tgt_elt_distri[0],          tgt_elt_distri[1])+1
  new_gnum = np.arange(tgt_n_elt+add_elt_distri[0],tgt_n_elt+add_elt_distri[1])+1
  elt_gnum = np.concatenate([old_gnum, new_gnum])

  cst_stride = np.full(elt_gnum.size, tgt_elt_size, np.int32)
  ptb = EP.PartToBlock(new_elt_distri, [elt_gnum], comm)
  _, tgt_new_ec = ptb.exchange_field([tgt_new_ec], part_stride=[cst_stride])
  PT.set_value(tgt_ec_n, tgt_new_ec)

  tgt_er = PT.Element.Range(tgt_elt_n)
  tgt_er[1] += add_elt_distri[2]

  apply_offset_to_elts(zone, add_elt_distri[2], tgt_er[1]-add_elt_distri[2])
  
  return new_gnum+tgt_elt_offset-1


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
      mask  = ~par_algo.gnum_isin(bc_pl, face_bc_pls, comm)

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

  elts = PT.get_nodes_from_label(zone, 'Elements_t')
  tri_elts   = [elt for elt in elts if PT.Element.CGNSName(elt)=='TRI_3']
  tetra_elts = [elt for elt in elts if PT.Element.CGNSName(elt)=='TETRA_4']
  bar_elts   = [elt for elt in elts if PT.Element.CGNSName(elt)=='BAR_2']
  assert len(tri_elts) == len(tetra_elts) == 1, f"Multiple elts nodes are not managed"
  assert len(bar_elts) <= 1, f"Multiple elts nodes are not managed"
  tri_elt   = tri_elts[0]
  tetra_elt = tetra_elts[0]
  bar_elt   = bar_elts[0] if len(bar_elts) > 0 else None

  new_vtx_nums = list()
  to_constrain_bcs = list()
  matching_bcs   = list()
  for i_per, gc_paths in enumerate(jn_pairs):

    gc_vtx_n = PT.get_node_from_path(tree, gc_paths[0])
    gc_vtx_pl  = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointList'     ))[0]
    gc_vtx_pld = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointListDonor'))[0]

    # > 1/ Defining the internal surface, that will be constrained in mesh adaptation
    cell_pl = tag_elmt_owning_vtx(tetra_elt, gc_vtx_pld, comm, elt_full=False) # Tetra made of at least one gc opp vtx
    face_pl = add_undefined_faces(zone, tetra_elt, cell_pl, tri_elt, comm) # ?
    vtx_pl  = elmt_pl_to_vtx_pl(zone, tetra_elt, cell_pl, comm) # Vertices ids of tetra belonging to cell_pl

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

    # maia.io.dist_tree_to_file(tree, f'OUTPUT/internal_surface_{i_per}.cgns', comm)

    # > 2/ Removing lines defined on join because they surely has their periodic on the other side
    # > Find BCs on GCs that will be deleted because they have their periodic twin
    # > For now only fully described BCs will be treated
    if bar_elt is not None:
      bar_to_rm_pl  = tag_elmt_owning_vtx(bar_elt, gc_vtx_pld, comm, elt_full=True) #Bar made of two gc opp vtx
      bar_twins_pl  = tag_elmt_owning_vtx(bar_elt, gc_vtx_pl , comm, elt_full=True) #Bar made of two gc vtx
      _matching_bcs =find_matching_bcs(zone, bar_elt, bar_to_rm_pl, bar_twins_pl, [gc_vtx_pld, gc_vtx_pl], comm)
      remove_elts_from_pl(zone, bar_elt, bar_to_rm_pl, comm)
    else:
      _matching_bcs = list()
    matching_bcs.append(_matching_bcs)


    # > 3/ Duplicate elts and vtx defined on internal created surface, and updating vtx numbering
    #      of elmts touching this surface
    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_pl
    to_update_face_pl = tag_elmt_owning_vtx(tri_elt, vtx_pl, comm, elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(bar_elt, vtx_pl, comm, elt_full=True)

    # > Ambiguous faces that contains all vtx but are not included in patch cells can be removed
    to_update_face_pl = find_shared_faces(tri_elt, to_update_face_pl, tetra_elt, cell_pl, comm)

    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}
  
    face_bc_name = f'tri_3_periodic_{i_per}'
    new_vtx_num = duplicate_elts(zone, tri_elt, face_pl, f'tri_3_periodic_{i_per}', elts_to_update, comm)
    to_constrain_bcs.append(face_bc_name)


    # > 4/ Apply periodic transformation to vtx and flowsol
    vtx_pl  = elmt_pl_to_vtx_pl(zone, tetra_elt, cell_pl, comm)
    perio_val = PT.GridConnectivity.periodic_values(PT.get_node_from_path(tree, gc_paths[1]))
    periodic = perio_val.asdict(snake_case=True)

    dist_transform.transform_affine_zone(zone, vtx_pl, comm, **periodic, apply_to_fields=True)

    # > 5/ Merge two GCs that are now overlaping
    bc_name1 = PT.path_tail(gc_paths[0])
    bc_name2 = PT.path_tail(gc_paths[1])
    vtx_match_num = [gc_vtx_pl, gc_vtx_pld]
    vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
    vtx_tag = np.arange(vtx_distri[0], vtx_distri[1], dtype=vtx_distri.dtype)+1
    old_to_new_vtx = merge_periodic_bc(zone, (bc_name1, bc_name2), vtx_tag, vtx_match_num, comm, keep_original=True)
    
    fake_vtx_distri = par_utils.dn_to_distribution(old_to_new_vtx.size, comm)
    for i_previous_per in range(0, i_per):
      new_vtx_nums[i_previous_per][0] = EP.block_to_part(old_to_new_vtx, fake_vtx_distri, [new_vtx_nums[i_previous_per][0]], comm)[0]
      new_vtx_nums[i_previous_per][1] = EP.block_to_part(old_to_new_vtx, fake_vtx_distri, [new_vtx_nums[i_previous_per][1]], comm)[0]
    
    new_vtx_num[0] = EP.block_to_part(old_to_new_vtx, fake_vtx_distri, [new_vtx_num[0]], comm)[0]
    new_vtx_num[1] = EP.block_to_part(old_to_new_vtx, fake_vtx_distri, [new_vtx_num[1]], comm)[0]
    new_vtx_nums.append(new_vtx_num)

    # > Set Vertex BC to preserve join infos
    pl_constraint = new_vtx_num[0].reshape((1,-1), order='F')
    new_bc_distrib = par_utils.dn_to_distribution(pl_constraint.size, comm)
    bc_n = PT.new_BC(name=f'vtx_constraint_{i_per}',
                     type='FamilySpecified',
                     point_list=pl_constraint,
                     loc='Vertex',
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':new_bc_distrib}, parent=bc_n)

    pl_periodic = new_vtx_num[1].reshape((1,-1), order='F')
    new_bc_distrib = par_utils.dn_to_distribution(pl_periodic.size, comm)
    bc_n = PT.new_BC(name=f'vtx_periodic_{i_per}',
                     type='FamilySpecified',
                     point_list=pl_periodic,
                     loc='Vertex',
                     family='PERIODIC',
                     parent=zone_bc_n)
    PT.maia.newDistribution({'Index':new_bc_distrib}, parent=bc_n)

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

  elts = PT.get_nodes_from_label(zone, 'Elements_t')
  bar_elts   = [elt for elt in elts if PT.Element.CGNSName(elt)=='BAR_2']
  tri_elts   = [elt for elt in elts if PT.Element.CGNSName(elt)=='TRI_3']
  tetra_elts = [elt for elt in elts if PT.Element.CGNSName(elt)=='TETRA_4']
  assert len(tri_elts) == len(tetra_elts) == 1, f"Multiple elts nodes are not managed"
  assert len(bar_elts) <= 1, f"Multiple elts nodes are not managed"
  tri_elt   = tri_elts[0]
  tetra_elt = tetra_elts[0]
  bar_elt   = bar_elts[0] if len(bar_elts) > 0 else None

  cell_elt_name = 'TETRA_4' if is_3d else 'TRI_3'
  face_elt_name = 'TRI_3'   if is_3d else 'BAR_2'
  edge_elt_name = 'BAR_2'   if is_3d else  None

  # > Removing old periodic patch
  i_per = n_periodicity-1
  for gc_paths, periodic_values in reversed(list(jn_pairs_and_values.items())): # reversed for future multiperiodic (use list() for py 3.7 support)
  
    # > 1/ Get elts and vtx for BCs out of feflo
    #      TODO: some TRI_3 should be avoided with find_shared_faces again
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    cell_bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_bc_pl = PT.get_value(PT.Subset.getPatch(cell_bc_n))[0]
    vtx_pl = elmt_pl_to_vtx_pl(zone, tetra_elt, cell_bc_pl, comm)

    still_here_gc_name  = gc_paths[0].split('/')[-1]
    to_retrieve_gc_name = gc_paths[1].split('/')[-1]
    bc_n = PT.get_child_from_name(zone_bc_n, still_here_gc_name)
    face_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]

    # > Defining which element related to created surface must be updated
    to_update_cell_pl = cell_bc_pl
    to_update_face_pl = tag_elmt_owning_vtx(tri_elt, vtx_pl, comm, elt_full=True)
    to_update_line_pl = tag_elmt_owning_vtx(bar_elt, vtx_pl, comm, elt_full=True)
    to_update_face_pl = find_shared_faces(tri_elt, to_update_face_pl, tetra_elt, cell_bc_pl, comm)
    elts_to_update = {'TETRA_4': to_update_cell_pl, 'TRI_3':to_update_face_pl, 'BAR_2':to_update_line_pl}

    # > 2/ Duplicate GC surface and update element connectivities in the patch
    duplicate_elts(zone, tri_elt, face_pl, to_retrieve_gc_name, elts_to_update, comm, elt_duplicate_bcs=bcs_to_retrieve[i_per])

    # > 3/ Deplace periodic patch to retrieve initial domain
    #      (vtx_pl is updated because has changed with surface duplication)
    cell_bc_name = f'tetra_4_periodic_{i_per}'
    bc_n = PT.get_child_from_name(zone_bc_n, cell_bc_name)
    cell_pl = PT.get_value(PT.Subset.getPatch(bc_n))[0]
    vtx_pl  = elmt_pl_to_vtx_pl(zone, tetra_elt, cell_pl, comm)
    periodic = periodic_values[0].asdict(True)
    dist_transform.transform_affine_zone(zone, vtx_pl, comm, **periodic, apply_to_fields=True)
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
