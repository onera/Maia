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


def add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, periodic_transfo, comm):
  elt_n = PT.get_node_from_name(zone, 'TRI_3.0')
  elt_conn_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  elt_conn    = PT.get_value(elt_conn_n)
  elt_size = 3 # TRI_3 for now

  n_elt_to_add = gc_elt_pl.size

  pelt_conn = -np.ones(n_elt_to_add*elt_size, dtype=np.int32)
  for isize in range(elt_size):
    pelt_conn[isize::elt_size] = elt_conn[elt_size*(gc_elt_pl-1)+isize]

  # > Find vtx that are not tagged in GC
  tag_pvtx     = np.isin(pelt_conn, gc_vtx_pld) # True where vtx is 
  gc_pvtx_pl1  = np.where(          tag_pvtx )[0] # Which vtx is in gc
  gc_pvtx_pl2  = np.where(np.invert(tag_pvtx))[0] # Which vtx is not in gc
  vtx_pl_toadd = np.unique(pelt_conn[gc_pvtx_pl2])

  # > Updating vtx numbering for periodic added elements
  n_vtx       = PT.Zone.n_vtx(zone)
  n_vtx_toadd = vtx_pl_toadd.size
  vtx_transfo = {k:v   for k, v in zip(gc_vtx_pld, gc_vtx_pl)}
  new_num_vtx = {k:v+1 for k, v in zip(vtx_pl_toadd, np.arange(n_vtx, n_vtx+n_vtx_toadd))}
  pelt_conn[gc_pvtx_pl1] = [vtx_transfo[i_vtx] for i_vtx in pelt_conn[gc_pvtx_pl1]]
  pelt_conn[gc_pvtx_pl2] = [new_num_vtx[i_vtx] for i_vtx in pelt_conn[gc_pvtx_pl2]]

  elt_n = PT.get_node_from_name(zone, 'TRI_3.0')
  elt_conn_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  elt_conn    = PT.get_value(elt_conn_n)
  elt_conn    = np.concatenate([elt_conn, pelt_conn])
  PT.set_value(elt_conn_n, elt_conn)
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
  pcx = cx[vtx_pl_toadd-1] - periodic_transfo[0]
  pcy = cy[vtx_pl_toadd-1] - periodic_transfo[1]
  pcz = cz[vtx_pl_toadd-1] - periodic_transfo[2]
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
        print(f'   DataArray: \"{PT.get_name(da_n)}\": {da.size} {vtx_pl_toadd.size}')
        da_to_add = da[vtx_pl_toadd-1]
        PT.set_value(da_n, np.concatenate([da, da_to_add]))

  # > Updating zone dimensions
  PT.set_value(zone, [[n_vtx+n_vtx_toadd, elt_range[1], 0]])
 


  old_vtx_num = np_utils.concatenate_np_arrays([np.array(list(vtx_transfo.keys  ()),dtype=np.int32),
                                                np.array(list(new_num_vtx.keys  ()),dtype=np.int32)])[1]
  new_vtx_num = np_utils.concatenate_np_arrays([np.array(list(vtx_transfo.values()),dtype=np.int32),
                                                np.array(list(new_num_vtx.values()),dtype=np.int32)])[1]
  add_bc_on_periodic_patch(zone, gc_elt_pl, old_vtx_num, new_vtx_num, 'BAR_2', comm)

  return n_vtx_toadd, new_num_vtx


def add_constraint_bcs(zone, new_num_vtx):
  n_vtx_toadd = len(new_num_vtx)

  bar_n = PT.get_node_from_name(zone, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range   = PT.get_value(bar_elt_range_n)
  bar_elt_conn_n = PT.get_child_from_name(bar_n, 'ElementConnectivity')
  bar_elt_conn   = PT.get_value(bar_elt_conn_n)

  n_bar = bar_elt_range[1]-bar_elt_range[0]+1

  for bar_to_add in [new_num_vtx.keys(), new_num_vtx.values()]:
    pbar_conn = -np.ones((n_vtx_toadd-1)*2, dtype=np.int32)
    pbar_conn[0::2] = np.array(list(bar_to_add), dtype=np.int32)[0:-1]
    pbar_conn[1::2] = np.array(list(bar_to_add), dtype=np.int32)[1:]
    bar_elt_range[1]= bar_elt_range[1]+(n_vtx_toadd-1)
    bar_elt_conn    = np.concatenate([bar_elt_conn, pbar_conn])

  PT.set_value(bar_elt_range_n, bar_elt_range)
  PT.set_value(bar_elt_conn_n, bar_elt_conn)

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

  pl_fixed = np.array(list(new_num_vtx.keys()), dtype=np.int32).reshape((1,-1), order='F')
  # pl_fixedp = np.arange(1, n_vtx_toadd+1, dtype=np.int32).reshape((1,-1), order='F') # not CGNS coherent, but tricks for periodic adaptation
  PT.new_BC(name='vtx_fixed',
            type='BCWall',
            point_list=pl_fixed,
            loc='Vertex',
            family='BCS',
            parent=zone_bc_n)
  pl_fixedp = np.array(list(new_num_vtx.values()), dtype=np.int32).reshape((1,-1), order='F')
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
  elt_size = PT.Element.NVtx(elt_n)

  # > Update connectivity
  er[1] = er[1]-n_elt_to_rm
  last_elt = er[1]
  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])
  pl_c  = -np.ones(n_elt_to_rm*elt_size, dtype=np.int32)
  for i_size in range(elt_size):
    pl_c[i_size::elt_size] = elt_size*(pl-offset-1)+i_size
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
  elt_size = PT.Element.NVtx(elt_n)

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
  print(f'old_to_new_vtx_num = {old_to_new_vtx_num}')

  pbc1_n = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_pl = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  pbc1_vtx_pl  = elmt_pl_to_vtx_pl(zone, pbc1_pl, 'BAR_2')
  print(f'pbc1_vtx_pl = {pbc1_vtx_pl} -> {vtx_tag[pbc1_vtx_pl-1]}')

  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  pbc2_vtx_pl  = elmt_pl_to_vtx_pl(zone, pbc2_pl, 'BAR_2')
  print(f'pbc2_vtx_pl = {pbc2_vtx_pl} -> {vtx_tag[pbc2_vtx_pl-1]}')

  old_vtx_num = np.flip(np.array(list(old_to_new_vtx_num.keys())  , dtype=np.int32)+1) # flip to debug, can be remove
  new_vtx_num = np.flip(np.array(list(old_to_new_vtx_num.values()), dtype=np.int32)+1)
  print(f'\n')
  print(f'old_vtx_num = {old_vtx_num}')
  print(f'new_vtx_num = {new_vtx_num}')


  pl1_tag = vtx_tag[pbc1_vtx_pl-1]
  sort_old = np.argsort(old_vtx_num)
  print(f'\n')

  print(f'searching = {pl1_tag}')
  print(f'   -> in  = {old_vtx_num}')
  print(f'sort_tab = {sort_old}')
  print(f'\n')
  idx_pl1_tag_in_old = np.searchsorted(old_vtx_num, pl1_tag, sorter=sort_old)
  print(f'idx_old_in_pl1_tag = {sort_old[idx_pl1_tag_in_old]}')
  print(f'old_vtx_num[sort_tab] = {old_vtx_num[sort_old[idx_pl1_tag_in_old]]}')


  pl2_tag = vtx_tag[pbc2_vtx_pl-1]
  sort_pl2_tag = np.argsort(pl2_tag)
  print(f'\n')
  print(f'searching = {new_vtx_num}')
  print(f'   -> in  = {pl2_tag}')
  print(f'sort_tab = {sort_pl2_tag}')
  idx_new_in_pl2_tag = np.searchsorted(pl2_tag, new_vtx_num, sorter=sort_pl2_tag)
  print(f'idx_old_in_pl2_tag = {sort_old[idx_pl1_tag_in_old]}')
  print(f'pl2_tag[sort_tab] = {pl2_tag[sort_pl2_tag[idx_new_in_pl2_tag]]}')

  print(f'\n')
  print(f'pbc2_vtx_pl = {pbc2_vtx_pl[sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]]}')
  print(f'pbc1_vtx_pl = {pbc1_vtx_pl}')

  sources = pbc2_vtx_pl[sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]]
  targets = pbc1_vtx_pl
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, sources, targets, comm, False)


  bar_to_rm_pl = tag_elmt_owning_vtx(zone, pbc2_vtx_pl, 'BAR_2', elt_full=True)
  print(f'bar_to_rm_pl = {bar_to_rm_pl.size}')
  print('Removing BAR_2 elements pl1')
  print(f'pbc1_pl = {pbc1_pl}')
  remove_elts_from_pl(zone, pbc1_pl, 'BAR_2', comm)
  print('Removing BAR_2 elements pl2')
  print(f'pbc2_pl = {pbc2_pl}')
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  print(f'pbc2_pl = {pbc2_pl}')
  remove_elts_from_pl(zone, pbc2_pl, 'BAR_2', comm)
  print('Updating vtx numbering')
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



def deplace_periodic_patch(zone, patch_name, gc_name, periodic_transfo, bc_to_update, comm):

  # > Get_infos
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  patch_n       = PT.get_child_from_name(zone_bc_n, patch_name)
  patch_pl      = PT.get_value(PT.get_child_from_name(patch_n, 'PointList'))[0]
  print(f'n_patch_tri = {patch_pl.size}')
  patch_vtx_pl  = elmt_pl_to_vtx_pl(zone, patch_pl, 'TRI_3')
  print(f'n_patch_vtx = {patch_vtx_pl.size}')

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
  print(f'tri_pl = {pl}')
  n_tri_gc = pl.size
  print(f'n_tri_gc = {n_tri_gc}')
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  conn_cp = ec[conn_pl]
  mask    = np.isin(conn_cp, pl_vtx_duplicate)

  conn_pl = conn_pl[mask]
  conn_cp = conn_cp[mask]
  print(f'conn_pl.size = {conn_pl.size}')

  print(f'pl_vtx_duplicate = {pl_vtx_duplicate}')
  print(f'isin = {np.isin(pl_vtx_duplicate, conn_cp)}')
  print(f'isin = {np.isin(conn_cp, pl_vtx_duplicate)}')
  new_conn = np.searchsorted(pl_vtx_duplicate, conn_cp) #pl_vtx_duplicate already sorted by unique
  new_conn = new_conn+n_vtx+1
  print(f'comm_cp  = {conn_cp}')
  print(f'new_conn = {new_conn}')
  ec[conn_pl] = new_conn
  PT.set_value(ec_n, ec)

  PT.set_value(zone, [[n_vtx+n_new_vtx, n_tri, 0]])

  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  PT.add_child(base, zone)
  import maia
  maia.io.write_tree(tree, 'OUTPUT/test.cgns')


  # > Deplace vtx that are not on GC
  print(f'tri_to_move = {pl+1}')
  print()
  patch_vtx_pl  = elmt_pl_to_vtx_pl(zone, patch_pl, 'TRI_3') #Connectivity has changed
  print(f'patch_vtx_pl = {patch_vtx_pl.size}')
  selected_vtx   = np.isin(patch_vtx_pl, gc_vtx_pl, invert=True)
  print(f'selected_vtx = {selected_vtx}')
  vtx_to_move_pl = patch_vtx_pl[selected_vtx]

  print(f'n_vtx_to_move = {vtx_to_move_pl.size}')
  print(f'vtx_to_move_pl = {vtx_to_move_pl}')

  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  cx[vtx_to_move_pl-1] = cx[vtx_to_move_pl-1] + periodic_transfo[0]
  cy[vtx_to_move_pl-1] = cy[vtx_to_move_pl-1] + periodic_transfo[1]
  cz[vtx_to_move_pl-1] = cz[vtx_to_move_pl-1] + periodic_transfo[2]
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
    print(f'[updating bar ec] BC \"{bc_name}\"')
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    bc_pl = bc_pl - offset -1
    ec_pl = np_utils.interweave_arrays([size_elt*bc_pl+i_size for i_size in range(size_elt)])
    ec_bc = ec[ec_pl]
    mask  = np.isin(ec_bc, pl_vtx_duplicate)
    ec_pl = ec_pl[mask]
    ec_bc = ec_bc[mask]
    
    new_ec = np.searchsorted(pl_vtx_duplicate, ec_bc)
    new_ec = new_ec+n_vtx+1
    print(f'   ec_bc = {ec_bc} -> {new_conn}')
    ec[ec_pl] = new_ec

    twin_bc_n    = PT.get_child_from_name(zone_bc_n, bc_name[:-1])
    twin_bc_pl_n = PT.get_child_from_name(twin_bc_n, 'PointList')
    twin_bc_pl = PT.get_value(twin_bc_pl_n)[0]
    bc_pl      = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
    print(f'pls = {bc_pl} {bc_pl}')
    PT.print_tree(twin_bc_n)
    PT.set_value(twin_bc_pl_n, np.concatenate([twin_bc_pl, bc_pl]).reshape((1,-1), order='F'))
    PT.print_tree(twin_bc_n)

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
  print(f'pl_vtx_duplicate = {pl_vtx_duplicate}')
  print(f'ec_duplicate = {ec_duplicate}')
  print(f'new_vtx_num = {new_conn}')

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
  print(f'new_bc_pl = {new_bc_pl}')
  PT.new_BC('Xmax',
            type='BCWall',
            point_list=new_bc_pl.reshape((1,-1), order='F'),
            loc=CGNS_TO_LOC[cgns_name],
            family='BCS',
            parent=zone_bc_n)
  


  return n_vtx_duplicate, n_elt_to_add, pl_vtx_duplicate