import copy

import maia
import maia.pytree        as PT
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
from   maia.utils         import np_utils

from maia.transfer import protocols as EP

from maia.algo.dist.merge_ids import merge_distributed_ids


import numpy as np

CGNS_TO_LOC = {'BAR_2':'EdgeCenter',
               'TRI_3':'FaceCenter'}

def elmt_pl_to_vtx_pl(zone, pl, cgns_name):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  pl = pl - local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]]) -1
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  vtx_pl = np.unique(ec[conn_pl])

  return vtx_pl


def tag_elmt_owning_vtx(zone, pl, cgns_name, elt_full=False):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  tag_vtx   = np.isin(ec, pl) # True where vtx is 
  if elt_full:
    tag_tri = np.logical_and.reduceat(tag_vtx, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  else:
    tag_tri = np.logical_or .reduceat(tag_vtx, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  gc_tri_pl = np.where(tag_tri)[0]+1 # Which cells has vtx

  return gc_tri_pl


def add_bc_on_periodic_patch(zone, elt_pl, old_vtx_num, new_vtx_num, cgns_name, comm):
  elt_vtx_pl = elmt_pl_to_vtx_pl(zone, elt_pl, 'TRI_3')

  # > Find elts having vtx in periodic patch vtx
  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  ec_n    = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec      = PT.get_value(ec_n)
  tag_elt = np.isin(ec, elt_vtx_pl)
  tag_elt = np.logical_and.reduceat(tag_elt, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  ppatch_elts_pl  = np.where(tag_elt)[0]+1

  is_asked_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_asked_bc):
    bc_name = PT.get_name(bc_n)
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    tag  = np.isin(bc_pl-offset, ppatch_elts_pl)
    new_bc_pl = bc_pl[tag]
    if new_bc_pl.size!=0:
      n_new_elt += new_bc_pl.size
      new_bcs[bc_name+'p'] = new_bc_pl
      PT.new_BC(bc_name+'p',
                type='BCWall',
                point_list=new_bc_pl.reshape((1,-1), order='F'),
                loc=CGNS_TO_LOC[cgns_name],
                family='BCS',
                parent=zone_bc_n)

  # > Building new elt connectivity
  new_elt_pl = np_utils.concatenate_np_arrays(new_bcs.values())[1]
  new_ec = -np.ones(n_new_elt*size_elt, dtype=np.int32)
  for isize in range(size_elt):
    new_ec[isize::size_elt] = ec[size_elt*(new_elt_pl-offset-1)+isize]

  n_vtx = PT.Zone.n_vtx(zone)
  old_to_new_vtx_num = -np.ones(n_vtx, dtype=np.int32)
  old_to_new_vtx_num[old_vtx_num-1] = new_vtx_num
  new_ec = old_to_new_vtx_num[new_ec-1]

  new_elt_num = np.arange(n_elt, n_elt+n_new_elt, dtype=np.int32)
  old_to_new_elt_num = -np.ones(n_elt, dtype=np.int32)
  old_to_new_elt_num[new_elt_pl-offset-1] = new_elt_num
  for bc_name in new_bcs.keys():
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    pl_n = PT.get_child_from_name(bc_n, 'PointList')
    pl = PT.get_value(pl_n)[0]
    new_pl = old_to_new_elt_num[pl-offset-1]+offset+1
    PT.set_value(pl_n, new_pl.reshape((1,-1), order='F'))

  # > Update connectivity
  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)
  ec   = np.concatenate([ec, new_ec])
  PT.set_value(ec_n, ec)

  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  er[1] = er[1] + n_new_elt
  PT.set_value(er_n, er)


def add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, periodic_values, cgns_name, comm):
  
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)

  size_elt = PT.Element.NVtx(elt_n)
  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])

  n_elt_to_add = gc_elt_pl.size

  # > Get elts connectivity
  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = gc_elt_pl - offset -1
  ec_pl  = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  ec_elt = ec[ec_pl]

  # > Find vtx that are tagged in GC or not
  tag_pvtx      = np.isin(ec_elt, gc_vtx_pld) # True where vtx is 
  gc_pvtx_pl1   = np.where(          tag_pvtx )[0] # Which vtx is in gc
  gc_pvtx_pl2   = np.where(np.invert(tag_pvtx))[0] # Which vtx is not in gc
  vtx_pl_to_add = np.unique(ec_elt[gc_pvtx_pl2])

  # > Updating vtx numbering for periodic added elements
  n_vtx       = PT.Zone.n_vtx(zone)
  n_vtx_toadd = vtx_pl_to_add.size
  vtx_transfo = [gc_vtx_pld, gc_vtx_pl]

  sort_vtx_transfo = np.argsort(vtx_transfo[0])
  idx_vtx_transfo = np.searchsorted(vtx_transfo[0], ec_elt[gc_pvtx_pl1], sorter=sort_vtx_transfo)



  # sys.exit()
  new_num_vtx = [vtx_pl_to_add, np.arange(n_vtx, n_vtx+n_vtx_toadd)+1]
  sort_new_num_vtx = np.argsort(new_num_vtx[0])
  idx_new_num_vtx = np.searchsorted(new_num_vtx[0], ec_elt[gc_pvtx_pl2], sorter=sort_new_num_vtx)
  
  # > Update connectivity with new vtx
  ec_elt[gc_pvtx_pl1] = vtx_transfo[1][sort_vtx_transfo[idx_vtx_transfo]]
  ec_elt[gc_pvtx_pl2] = new_num_vtx[1][sort_new_num_vtx[idx_new_num_vtx]]

  # > Update element node
  ec    = np.concatenate([ec, ec_elt])
  PT.set_value(ec_n, ec)
  elt_range_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  elt_range    = PT.get_value(elt_range_n)
  elt_range[1] = elt_range[1]+n_elt_to_add
  PT.set_value(elt_range_n, elt_range)


  # > Updating offset others elements
  bar_n = PT.get_node_from_name(zone, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range = PT.get_value(bar_elt_range_n)
  bar_offset = elt_range[1]-bar_elt_range[0]+1
  bar_elt_range[0] = bar_elt_range[0]+bar_offset
  bar_elt_range[1] = bar_elt_range[1]+bar_offset
  PT.set_value(bar_elt_range_n, bar_elt_range)

  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and\
                         PT.Subset.GridLocation(n)=='EdgeCenter'
  for edge_bc_n in PT.get_nodes_from_predicate(zone, is_edge_bc):
    pl_n = PT.get_child_from_name(edge_bc_n, 'PointList')
    pl = PT.get_value(pl_n)
    PT.set_value(pl_n, pl+bar_offset)
  
  # > Updating coordinates with duplicated vtx
  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  pcx = -np.ones(n_vtx_toadd, dtype=np.float64)
  pcy = -np.ones(n_vtx_toadd, dtype=np.float64)
  pcz = -np.ones(n_vtx_toadd, dtype=np.float64)
  pcx = cx[vtx_pl_to_add-1] - periodic_values[2][0] # Only translation for now TODO
  pcy = cy[vtx_pl_to_add-1] - periodic_values[2][1] # Only translation for now TODO
  pcz = cz[vtx_pl_to_add-1] - periodic_values[2][2] # Only translation for now TODO
  PT.set_value(cx_n, np.concatenate([cx, pcx]))
  PT.set_value(cy_n, np.concatenate([cy, pcy]))
  PT.set_value(cz_n, np.concatenate([cz, pcz]))


  # > Update flow_sol
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    print(f'FlowSolution: \"{PT.get_name(fs_n)}\"')
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      if PT.get_name(da_n)!='vtx_tag':
        da = PT.get_value(da_n)
        print(f'   DataArray: \"{PT.get_name(da_n)}\": {da.size} {vtx_pl_to_add.size}')
        da_to_add = da[vtx_pl_to_add-1]
        PT.set_value(da_n, np.concatenate([da, da_to_add]))

  # > Updating zone dimensions
  PT.set_value(zone, [[n_vtx+n_vtx_toadd, elt_range[1], 0]])
 


  old_vtx_num = np_utils.concatenate_np_arrays([vtx_transfo[0], new_num_vtx[0]])[1]
  new_vtx_num = np_utils.concatenate_np_arrays([vtx_transfo[1], new_num_vtx[1]])[1]
  add_bc_on_periodic_patch(zone, gc_elt_pl, old_vtx_num, new_vtx_num, 'BAR_2', comm)

  return n_vtx_toadd, new_num_vtx


def add_constraint_bcs(zone, new_num_vtx):
  n_vtx_toadd = new_num_vtx[0].size

  bar_n = PT.get_node_from_name(zone, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range   = PT.get_value(bar_elt_range_n)
  bar_ec_n = PT.get_child_from_name(bar_n, 'ElementConnectivity')
  bar_ec   = PT.get_value(bar_ec_n)

  n_bar = bar_elt_range[1]-bar_elt_range[0]+1

  for bar_to_add in new_num_vtx:
    pbar_conn = -np.ones((n_vtx_toadd-1)*2, dtype=np.int32)
    # TODO: use interweave_array
    pbar_conn[0::2] = bar_to_add[0:-1]
    pbar_conn[1::2] = bar_to_add[1:]
    bar_elt_range[1]= bar_elt_range[1]+(n_vtx_toadd-1)
    bar_ec    = np.concatenate([bar_ec, pbar_conn])

  PT.set_value(bar_elt_range_n, bar_elt_range)
  PT.set_value(bar_ec_n, bar_ec)

  zone_bc_n = PT.get_node_from_label(zone, 'ZoneBC_t')
  PT.new_BC(name='fixed',
              type='BCWall',
              point_list=np.arange(bar_elt_range[0]+n_bar, bar_elt_range[0]+n_bar+(n_vtx_toadd-1), dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)
  PT.new_BC(name='fixedp',
              type='BCWall',
              point_list=np.arange(bar_elt_range[0]+n_bar+(n_vtx_toadd-1), bar_elt_range[0]+n_bar+(n_vtx_toadd-1)*2, dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)

  pl_fixed = new_num_vtx[0].reshape((1,-1), order='F')
  # pl_fixedp = np.arange(1, n_vtx_toadd+1, dtype=np.int32).reshape((1,-1), order='F') # not CGNS coherent, but tricks for periodic adaptation
  PT.new_BC(name='vtx_fixed',
            type='BCWall',
            point_list=pl_fixed,
            loc='Vertex',
            family='BCS',
            parent=zone_bc_n)
  pl_fixedp = new_num_vtx[1].reshape((1,-1), order='F')
  # pl_fixedp = np.arange(1, n_vtx_toadd+1, dtype=np.int32).reshape((1,-1), order='F') # not CGNS coherent, but tricks for periodic adaptation
  PT.new_BC(name='vtx_fixedp',
            type='BCWall',
            point_list=pl_fixedp,
            loc='Vertex',
            family='BCS',
            parent=zone_bc_n)


def update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, cgns_name, comm):
  
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  part_data = EP.block_to_part(old_to_new_vtx, vtx_distri_ini, [ec], comm)
  ec = part_data[0]

  PT.set_value(ec_n, ec)


def remove_elts_from_pl(zone, pl, cgns_name, comm):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)
  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  
  n_elt = er[1]-er[0]+1
  n_elt_to_rm = pl.size
  
  elt_dim  = PT.Element.Dimension(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  # > Update connectivity
  er[1] = er[1]-n_elt_to_rm
  last_elt = er[1]
  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])
  pl_c  = -np.ones(n_elt_to_rm*size_elt, dtype=np.int32)
  for i_size in range(size_elt):
    pl_c[i_size::size_elt] = size_elt*(pl-offset-1)+i_size
  ec = np.delete(ec, pl_c)

  PT.set_value(ec_n, ec)
  PT.set_value(er_n, er)

  # > Update BC PointList
  targets = -np.ones(pl.size, dtype=np.int32)
  elt_distri_ini = np.array([0,n_elt,n_elt], dtype=np.int32) # TODO pdm_gnum
  old_to_new_elt = merge_distributed_ids(elt_distri_ini, pl-offset, targets, comm, True)

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl = PT.get_value(bc_pl_n)[0]
    bc_pl = bc_pl-offset

    in_pl = np.isin(bc_pl, pl-offset)
    new_bc_pl = old_to_new_elt[bc_pl[np.invert(in_pl)]-1]
    tag_invalid_elt  = np.isin(new_bc_pl,-1)
    new_bc_pl = new_bc_pl[np.invert(tag_invalid_elt)]
    new_bc_pl = new_bc_pl+offset

    if new_bc_pl.size==0:
      print(f'BC \"{PT.get_name(bc_n)}\" is empty, node is deleted.')
      PT.rm_child(zone_bc_n, bc_n)
    else:
      PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))

  return n_elt-n_elt_to_rm


def apply_offset_to_elt(zone, offset, cgns_name):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  er    = er-offset
  PT.set_value(er_n, er)

  elt_dim  = PT.Element.Dimension(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  # > Update BC PointList with offset
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl = PT.get_value(bc_pl_n)[0]
    new_bc_pl = bc_pl - offset
    PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))


def find_invalid_elts(zone, cgns_name):

  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  tag_elt  = np.isin(ec,-1)
  tag_elt  = np.logical_or.reduceat(tag_elt, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  invalid_elts_pl  = np.where(tag_elt)[0]+1

  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])

  return invalid_elts_pl+offset


def merge_periodic_bc(zone, bc_names, vtx_tag, old_to_new_vtx_num, comm):
  n_vtx = PT.Zone.n_vtx(zone)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  pbc1_n = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_pl = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  pbc1_vtx_pl  = elmt_pl_to_vtx_pl(zone, pbc1_pl, 'BAR_2')

  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  pbc2_vtx_pl  = elmt_pl_to_vtx_pl(zone, pbc2_pl, 'BAR_2')

  old_vtx_num = np.flip(old_to_new_vtx_num[0]+1) # flip to debug, can be remove
  new_vtx_num = np.flip(old_to_new_vtx_num[1]+1)


  pl1_tag = vtx_tag[pbc1_vtx_pl-1]
  sort_old = np.argsort(old_vtx_num)
  idx_pl1_tag_in_old = np.searchsorted(old_vtx_num, pl1_tag, sorter=sort_old)


  pl2_tag = vtx_tag[pbc2_vtx_pl-1]
  sort_pl2_tag = np.argsort(pl2_tag)
  idx_new_in_pl2_tag = np.searchsorted(pl2_tag, new_vtx_num, sorter=sort_pl2_tag)

  sources = pbc2_vtx_pl[sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]]
  targets = pbc1_vtx_pl
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, sources, targets, comm, False)


  bar_to_rm_pl = tag_elmt_owning_vtx(zone, pbc2_vtx_pl, 'BAR_2', elt_full=True)
  remove_elts_from_pl(zone, pbc1_pl, 'BAR_2', comm)
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  remove_elts_from_pl(zone, pbc2_pl, 'BAR_2', comm)
  update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, 'TRI_3', comm)
  update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, 'BAR_2', comm)

  n_vtx_to_rm = pbc2_vtx_pl.size
  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  PT.set_value(cx_n, np.delete(cx, pbc2_vtx_pl-1))
  PT.set_value(cy_n, np.delete(cy, pbc2_vtx_pl-1))
  PT.set_value(cz_n, np.delete(cz, pbc2_vtx_pl-1))


  # > Update flow_sol
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    print(f'FlowSolution: \"{PT.get_name(fs_n)}\"')
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      if PT.get_name(da_n)!='vtx_tag':
        da = PT.get_value(da_n)
        print(f'   DataArray: \"{PT.get_name(da_n)}\": {da.size}')
        da = np.delete(da, pbc2_vtx_pl-1)
        PT.set_value(da_n, da)


  n_tri = PT.Zone.n_cell(zone)
  PT.set_value(zone, [[n_vtx-n_vtx_to_rm, n_tri, 0]])

  # PT.rm_child(zone_bc_n, pbc1_n)
  # PT.rm_child(zone_bc_n, pbc2_n)



def deplace_periodic_patch(zone, patch_name, gc_name, periodic_values, bc_to_update, comm):

  # > Get_infos
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  patch_n       = PT.get_child_from_name(zone_bc_n, patch_name)
  patch_pl      = PT.get_value(PT.get_child_from_name(patch_n, 'PointList'))[0]
  patch_vtx_pl  = elmt_pl_to_vtx_pl(zone, patch_pl, 'TRI_3')

  gc_n      = PT.get_child_from_name(zone_bc_n, gc_name)
  gc_pl     = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_vtx_pl = elmt_pl_to_vtx_pl(zone, gc_pl, 'BAR_2')


  # > Duplicate GC
  n_new_vtx, n_elt_vtx, pl_vtx_duplicate = duplicate_elts(zone, gc_pl, 'BAR_2', comm)


  # > Updating TRI_3 connectivity
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)=='TRI_3'
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_vtx = PT.Zone.n_vtx(zone)
  n_tri    = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  tri_gc_pl = tag_elmt_owning_vtx(zone, gc_vtx_pl, 'TRI_3', elt_full=False)
  select_tri = np.isin(tri_gc_pl, patch_pl)
  pl = tri_gc_pl[select_tri]-1
  n_tri_gc = pl.size
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  conn_cp = ec[conn_pl]
  mask    = np.isin(conn_cp, pl_vtx_duplicate)

  conn_pl = conn_pl[mask]
  conn_cp = conn_cp[mask]

  new_conn = np.searchsorted(pl_vtx_duplicate, conn_cp) #pl_vtx_duplicate already sorted by unique
  new_conn = new_conn+n_vtx+1
  ec[conn_pl] = new_conn
  PT.set_value(ec_n, ec)

  PT.set_value(zone, [[n_vtx+n_new_vtx, n_tri, 0]])

  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  PT.add_child(base, zone)


  # > Deplace vtx that are not on GC
  patch_vtx_pl  = elmt_pl_to_vtx_pl(zone, patch_pl, 'TRI_3') #Connectivity has changed
  selected_vtx   = np.isin(patch_vtx_pl, gc_vtx_pl, invert=True)
  vtx_to_move_pl = patch_vtx_pl[selected_vtx]


  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  cx[vtx_to_move_pl-1] = cx[vtx_to_move_pl-1] + periodic_values[2][0] # Only translation managed for now TODO
  cy[vtx_to_move_pl-1] = cy[vtx_to_move_pl-1] + periodic_values[2][1] # Only translation managed for now TODO
  cz[vtx_to_move_pl-1] = cz[vtx_to_move_pl-1] + periodic_values[2][2] # Only translation managed for now TODO
  PT.set_value(cx_n, cx)
  PT.set_value(cy_n, cy)
  PT.set_value(cz_n, cz)


  # > Update BCs from periodic patch
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)=='BAR_2'
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)
  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC['BAR_2']])

  size_elt = PT.Element.NVtx(elt_n)

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  for bc_name in bc_to_update:
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    bc_name = PT.get_name(bc_n)
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    bc_pl = bc_pl - offset -1
    ec_pl = np_utils.interweave_arrays([size_elt*bc_pl+i_size for i_size in range(size_elt)])
    ec_bc = ec[ec_pl]
    mask  = np.isin(ec_bc, pl_vtx_duplicate)
    ec_pl = ec_pl[mask]
    ec_bc = ec_bc[mask]
    
    new_ec = np.searchsorted(pl_vtx_duplicate, ec_bc)
    new_ec = new_ec+n_vtx+1
    ec[ec_pl] = new_ec

    twin_bc_n    = PT.get_child_from_name(zone_bc_n, bc_name[:-1])
    twin_bc_pl_n = PT.get_child_from_name(twin_bc_n, 'PointList')
    twin_bc_pl = PT.get_value(twin_bc_pl_n)[0]
    bc_pl      = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
    PT.set_value(twin_bc_pl_n, np.concatenate([twin_bc_pl, bc_pl]).reshape((1,-1), order='F'))

    PT.rm_child(zone_bc_n, bc_n)




  # > Update flow_sol
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(zone, is_vtx_fs):
    print(f'FlowSolution: \"{PT.get_name(fs_n)}\"')
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      # if PT.get_name(da_n)!='vtx_tag':
      da = PT.get_value(da_n)
      print(f'   DataArray: \"{PT.get_name(da_n)}\": {da.size}')
      da_to_add = da[pl_vtx_duplicate-1]
      PT.set_value(da_n, np.concatenate([da, da_to_add]))



  PT.set_value(ec_n, ec)


def duplicate_elts(zone, elt_pl, cgns_name, comm):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_elt    = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  n_elt_to_add = elt_pl.size

  pl = elt_pl - local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]]) -1
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  ec_duplicate = ec[conn_pl]


  # > Update connectivity with new vtx numbering
  n_vtx = PT.Zone.n_vtx(zone)
  pl_vtx_duplicate = np.unique(ec_duplicate)
  n_vtx_duplicate = pl_vtx_duplicate.size

  new_conn = np.searchsorted(pl_vtx_duplicate, ec_duplicate) #pl_vtx_duplicate already sorted by unique
  new_conn = new_conn +n_vtx+1

  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  dcx = cx[pl_vtx_duplicate-1]
  dcy = cy[pl_vtx_duplicate-1]
  dcz = cz[pl_vtx_duplicate-1]
  cx = np.concatenate([cx, dcx])
  cy = np.concatenate([cy, dcy])
  cz = np.concatenate([cz, dcz])
  PT.set_value(cx_n, cx)
  PT.set_value(cy_n, cy)
  PT.set_value(cz_n, cz)


  ec = np.concatenate([ec, new_conn])
  PT.set_value(ec_n, ec)
  er_n = PT.get_child_from_name(elt_n, 'ElementRange')
  er   = PT.get_value(er_n)
  er[1] = er[1] + n_elt_to_add
  PT.set_value(er_n, er)

  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  
  new_bc_pl = np.arange(er[1]-n_elt_to_add+1, er[1]+1, dtype=np.int32)
  PT.new_BC('Xmax',
            type='BCWall',
            point_list=new_bc_pl.reshape((1,-1), order='F'),
            loc=CGNS_TO_LOC[cgns_name],
            family='BCS',
            parent=zone_bc_n)
  


  return n_vtx_duplicate, n_elt_to_add, pl_vtx_duplicate



def duplicate_periodic_patch(dist_tree, gc_name, comm):

  zones = PT.get_nodes_from_label(dist_tree, 'Zone_t')
  assert len(zones)==1
  zone = zones[0]
  n_tri_old = PT.Zone.n_cell(zone)

  # > Get GCs info
  zone_gc_n = PT.get_child_from_label(zone, 'ZoneGridConnectivity_t')
  gc_n = PT.get_child_from_name(zone_gc_n, gc_name)
  assert PT.Subset.GridLocation(gc_n)=='EdgeCenter' # 2d for now
  periodic_values = PT.GridConnectivity.periodic_values(gc_n)

  gc_pl  = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_pld = PT.get_value(PT.get_child_from_name(gc_n, 'PointListDonor'))[0]
  gc_vtx_pl  = elmt_pl_to_vtx_pl(zone, gc_pl , 'BAR_2')
  gc_vtx_pld = elmt_pl_to_vtx_pl(zone, gc_pld, 'BAR_2')
  



  # > Add GCs as BCs
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  for gc_n in PT.get_nodes_from_label(zone_gc_n, 'GridConnectivity_t'):
    gc_name = PT.get_name(gc_n)
    gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
    PT.new_BC(name=gc_name,
              type='BCWall',
              point_list=gc_pl,
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)


  # > Add periodic elts connected to GC
  gc_elt_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, 'TRI_3', elt_full=False)
  n_vtx_toadd, new_num_vtx = add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, periodic_values, 'TRI_3', comm)


  # > Add volumic BCs so that we can delete patches after TODO: better way ?
  pl_vol  = np.arange(1, n_tri_old+1, dtype=np.int32)
  pl_vol  = np.delete(pl_vol, gc_elt_pl-1).reshape((1,-1), order='F')
  pl_volp = np.arange(n_tri_old, n_tri_old+gc_elt_pl.size+1, dtype=np.int32)
  pl_volp = pl_volp.reshape((1,-1), order='F')
  pl_volc = (gc_elt_pl).reshape((1,-1), order='F')
  PT.new_BC(name='vol',            type='BCWall', point_list=pl_vol , loc='FaceCenter', family='BCS', parent=zone_bc_n)
  PT.new_BC(name='vol_periodic',   type='BCWall', point_list=pl_volp, loc='FaceCenter', family='BCS', parent=zone_bc_n)
  PT.new_BC(name='vol_constraint', type='BCWall', point_list=pl_volc, loc='FaceCenter', family='BCS', parent=zone_bc_n)


  # > Create new BAR_2 elmts and associated BCs to constrain mesh adaptation
  add_constraint_bcs(zone, new_num_vtx)

  PT.rm_nodes_from_name(dist_tree, 'ZoneGridConnectivity')
  

  return periodic_values, new_num_vtx


def retrieve_initial_domain(dist_tree, periodic_values, new_num_vtx, comm):

  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  n_vtx = PT.Zone.n_vtx(dist_zone)

  zone_bc_n = PT.get_child_from_label(dist_zone, 'ZoneBC_t')


  # > Removing old periodic patch
  bc_to_rm = PT.get_child_from_name(zone_bc_n, 'vol_constraint')
  bc_to_rm_pl = PT.get_value(PT.get_child_from_name(bc_to_rm, 'PointList'))[0]
  bc_to_rm_vtx_pl  = elmt_pl_to_vtx_pl(dist_zone, bc_to_rm_pl, 'TRI_3')
  n_elt_to_rm = bc_to_rm_pl.size

  bc_to_keep = PT.get_child_from_name(zone_bc_n, 'fixed')
  bc_to_keep_pl = PT.get_value(PT.get_child_from_name(bc_to_keep, 'PointList'))[0]
  bc_to_keep_vtx_pl = elmt_pl_to_vtx_pl(dist_zone, bc_to_keep_pl, 'BAR_2')
  n_vtx_to_keep = bc_to_keep_vtx_pl.size


  tag_vtx = np.isin(bc_to_rm_vtx_pl, bc_to_keep_vtx_pl) # True where vtx is 
  preserved_vtx_id = bc_to_rm_vtx_pl[tag_vtx][0]
  bc_to_rm_vtx_pl = bc_to_rm_vtx_pl[np.invert(tag_vtx)]
  n_vtx_to_rm = bc_to_rm_vtx_pl.size


  # > Compute new vtx numbering
  vtx_tag_n = PT.get_node_from_name(dist_zone, 'vtx_tag')
  vtx_tag   = PT.get_value(vtx_tag_n)
  vtx_tag = np.delete(vtx_tag, bc_to_rm_vtx_pl-1)
  PT.set_value(vtx_tag_n, vtx_tag)

  bc_fixed_n = PT.get_node_from_name_and_label(dist_zone, 'fixed', 'BC_t')
  bc_fixed_pl = PT.get_value(PT.get_child_from_name(bc_fixed_n, 'PointList'))[0]
  bc_fixed_vtx_pl = elmt_pl_to_vtx_pl(dist_zone, bc_fixed_pl, 'BAR_2')

  bc_fixedp_n = PT.get_node_from_name_and_label(dist_zone, 'fixedp', 'BC_t')
  bc_fixedp_pl = PT.get_value(PT.get_child_from_name(bc_fixedp_n, 'PointList'))[0]
  bc_fixedp_vtx_pl = elmt_pl_to_vtx_pl(dist_zone, bc_fixedp_pl, 'BAR_2')

  ids = bc_to_rm_vtx_pl
  targets = -np.ones(bc_to_rm_vtx_pl.size, dtype=np.int32)
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, ids, targets, comm, True)

  update_elt_vtx_numbering(dist_zone, vtx_distri_ini, old_to_new_vtx, 'TRI_3', comm)
  n_tri = remove_elts_from_pl(dist_zone, bc_to_rm_pl, 'TRI_3', comm)
  
  tri_offset = bc_to_rm_pl.size
  apply_offset_to_elt(dist_zone, tri_offset, 'BAR_2')
  update_elt_vtx_numbering(dist_zone, vtx_distri_ini, old_to_new_vtx, 'BAR_2', comm)
  invalid_elt_pl = find_invalid_elts(dist_zone, 'BAR_2')
  n_bar = remove_elts_from_pl(dist_zone, invalid_elt_pl, 'BAR_2', comm)

  cx_n = PT.get_node_from_name(dist_zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(dist_zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(dist_zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  PT.set_value(cx_n, np.delete(cx, bc_to_rm_vtx_pl-1))
  PT.set_value(cy_n, np.delete(cy, bc_to_rm_vtx_pl-1))
  PT.set_value(cz_n, np.delete(cz, bc_to_rm_vtx_pl-1))


  # > Update flow_sol
  is_vtx_fs = lambda n: PT.get_label(n)=='FlowSolution_t' and\
                        PT.Subset.GridLocation(n)=='Vertex'
  for fs_n in PT.get_children_from_predicate(dist_zone, is_vtx_fs):
    print(f'FlowSolution: \"{PT.get_name(fs_n)}\"')
    for da_n in PT.get_children_from_label(fs_n, 'DataArray_t'):
      if PT.get_name(da_n)!='vtx_tag':
        da = PT.get_value(da_n)
        print(f'   DataArray: \"{PT.get_name(da_n)}\": {da.size}')
        da = np.delete(da, bc_to_rm_vtx_pl-1)
        PT.set_value(da_n, da)

  PT.set_value(dist_zone, [[n_vtx-n_vtx_to_rm, n_tri, 0]])



  # > Deplace periodic patch to retrieve initial domain
  deplace_periodic_patch(dist_zone, 'vol_periodic', 'Xmin', periodic_values, ['Yminp', 'Ymaxp'], comm)
  merge_periodic_bc(dist_zone, ['fixed', 'fixedp'], vtx_tag, new_num_vtx, comm)

