import copy

import maia
import maia.pytree        as PT
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
from   maia.utils         import np_utils

from maia.transfer import protocols as EP

from maia.algo.dist.merge_ids import merge_distributed_ids


import numpy as np

CGNS_TO_LOC = {'BAR_2'  :'EdgeCenter',
               'TRI_3'  :'FaceCenter',
               'TETRA_4':'CellCenter'}
# LOC_TO_CGNS = {'EdgeCenter':'BAR_2',
#                'FaceCenter':'TRI_3',
#                'CellCenter':'TETRA_4'}

def elmt_pl_to_vtx_pl(zone, pl, cgns_name):
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec  = PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity'))
  elt_offset = PT.Element.Range(elt_n)[0]

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  pl = pl - elt_offset
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

  tag_vtx = np.isin(ec, pl) # True where vtx is
  if elt_full:
    tag_tri = np.logical_and.reduceat(tag_vtx, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  else:
    tag_tri = np.logical_or .reduceat(tag_vtx, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  gc_tri_pl = np.where(tag_tri)[0]+1 # Which cells has vtx

  return gc_tri_pl


def report_bcs_on_periodic_patch(zone, elt_n, elt_vtx_pl, old_vtx_num, new_vtx_num, comm):

  # > Find elts having vtx in periodic patch vtx
  # offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])
  # is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
  #                          PT.Element.CGNSName(n)==cgns_name
  # elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  
  elt_dim = PT.Element.Dimension(elt_n)
  elt_cgnsname = PT.Element.CGNSName(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  print(f'elt_cgnsname = {elt_cgnsname}')
  print(f'  -> elt_dim = {elt_dim}')

  ec_n    = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec      = PT.get_value(ec_n)
  tag_elt = np.isin(ec, elt_vtx_pl)
  tag_elt = np.logical_and.reduceat(tag_elt, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  ppatch_elts_pl  = np.where(tag_elt)[0]+1
  print(f'  -> n_elt_to_add = {ppatch_elts_pl.size}')

  check_if_elt_included(ppatch_elts_pl)  


  sys.exit()
  is_asked_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[elt_cgnsname]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_asked_bc):
    bc_name = PT.get_name(bc_n)
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    tag  = np.isin(bc_pl-elt_offset, ppatch_elts_pl)
    new_bc_pl = bc_pl[tag]
    if new_bc_pl.size!=0:
      print(f'    -> bc to add {bc_name}: new_bc_pl.size = {new_bc_pl.size}')
      n_new_elt += new_bc_pl.size
      new_bcs[bc_name+'p'] = new_bc_pl
      PT.new_BC(bc_name+'p',
                type='FamilySpecified',
                point_list=new_bc_pl.reshape((1,-1), order='F'),
                loc=CGNS_TO_LOC[elt_cgnsname],
                family='PERIODIC',
                parent=zone_bc_n)

  # > Building new elt connectivity
  new_elt_pl = np_utils.concatenate_np_arrays(new_bcs.values())[1]
  new_ec = -np.ones(n_new_elt*size_elt, dtype=np.int32)
  for isize in range(size_elt):
    new_ec[isize::size_elt] = ec[size_elt*(new_elt_pl-elt_offset-1)+isize]

  # print(f'old_vtx_num = {old_vtx_num}')
  # print(f'new_vtx_num = {new_vtx_num}')
  n_vtx = PT.Zone.n_vtx(zone)
  old_to_new_vtx_num = -np.ones(n_vtx, dtype=np.int32)
  old_to_new_vtx_num[old_vtx_num-1] = new_vtx_num
  new_ec = old_to_new_vtx_num[new_ec-1]

  new_elt_num = np.arange(n_elt, n_elt+n_new_elt, dtype=np.int32)
  old_to_new_elt_num = -np.ones(n_elt, dtype=np.int32)
  old_to_new_elt_num[new_elt_pl-elt_offset-1] = new_elt_num
  for bc_name in new_bcs.keys():
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    pl_n = PT.get_child_from_name(bc_n, 'PointList')
    pl = PT.get_value(pl_n)[0]
    new_pl = old_to_new_elt_num[pl-elt_offset-1]+elt_offset+1
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


def add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, new_vtx_num, periodic_values, cgns_name, comm):
  '''

  '''
  # print(f'gc_elt_pl = {gc_elt_pl}')

  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  dim_elt = PT.Element.Dimension(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]
  n_elt = PT.Element.Size(elt_n)
  # print(f"   - elt_name={PT.get_name(elt_n)}")
  # print(f'   - LOC_TO_DIM[CGNS_TO_LOC[cgns_name]] = {LOC_TO_DIM[CGNS_TO_LOC[cgns_name]]}')
  # print(f"   - elt_offset={elt_offset}")
  PT.print_tree(elt_n)

  # > Get elts connectivity
  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = gc_elt_pl -1
  ec_pl  = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  ec_elt = ec[ec_pl]


  # Filter elt with all vtx in donor gc
  n_elt_to_add = gc_elt_pl.size
  tag_elt = np.isin(ec_elt, gc_vtx_pld)
  tag_elt = np.logical_and.reduceat(tag_elt, np.arange(0,n_elt_to_add*size_elt,size_elt)) # True when has vtx 
  gc_elt_pl = gc_elt_pl[np.invert(tag_elt)]
  n_elt_to_add = gc_elt_pl.size
  new_elt_pl = np.arange(n_elt+1, n_elt+n_elt_to_add+1, dtype=np.int32)
  pl     = gc_elt_pl -1
  ec_pl  = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  ec_elt = ec[ec_pl]
  

  # > Find vtx that are tagged in GC or not
  tag_pvtx      = np.isin(ec_elt, gc_vtx_pld) # True where vtx is 
  gc_pvtx_pl1   = np.where(          tag_pvtx )[0] # Which vtx is in gc
  gc_pvtx_pl2   = np.where(np.invert(tag_pvtx))[0] # Which vtx is not in gc
  vtx_pl_to_add = np.unique(ec_elt[gc_pvtx_pl2])

  # > Updating vtx numbering for periodic added elements
  n_vtx       = PT.Zone.n_vtx(zone)

  vtx_match = [gc_vtx_pld, gc_vtx_pl]
  sort_vtx_match = np.argsort(vtx_match[0])
  idx_vtx_match = np.searchsorted(vtx_match[0], ec_elt[gc_pvtx_pl1], sorter=sort_vtx_match)


  tag_vtx_to_add = np.isin(vtx_pl_to_add, new_vtx_num[0], invert=True)
  # print(f'new_vtx_num[0] = {new_vtx_num[0]}')
  # print(f'vtx_pl_to_add  = {vtx_pl_to_add}')
  # print(f'tag_vtx_to_add = {tag_vtx_to_add}')
  vtx_pl_to_add = vtx_pl_to_add[tag_vtx_to_add]
  # print(f'vtx_pl_to_add  = {vtx_pl_to_add}')
  n_vtx_toadd = vtx_pl_to_add.size
  # print(f'   - n_vtx_toadd = {n_vtx_toadd}')
  if vtx_pl_to_add.size!=0:
    new_vtx_num[0] = np.concatenate([new_vtx_num[0], vtx_pl_to_add])
    new_vtx_num[1] = np.concatenate([new_vtx_num[1], np.arange(n_vtx, n_vtx+n_vtx_toadd)+1])
  sort_new_vtx_num = np.argsort(new_vtx_num[0])
  idx_new_vtx_num = np.searchsorted(new_vtx_num[0], ec_elt[gc_pvtx_pl2], sorter=sort_new_vtx_num)
  
  # > Update connectivity with new vtx
  ec_elt[gc_pvtx_pl1] = vtx_match[1][sort_vtx_match[idx_vtx_match]]
  ec_elt[gc_pvtx_pl2] = new_vtx_num[1][sort_new_vtx_num[idx_new_vtx_num]]

  # > Update element node
  ec = np.concatenate([ec, ec_elt])
  PT.set_value(ec_n, ec)
  elt_range_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  elt_range    = PT.get_value(elt_range_n)
  elt_range[1] = elt_range[1]+n_elt_to_add
  PT.set_value(elt_range_n, elt_range)

  

  # > Updating offset others elements
  elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  for dim in range(dim_elt-1,0,-1):
    assert len(elts_per_dim[dim])
    infdim_elt_n = elts_per_dim[dim][0]
    infdim_elt_range_n = PT.get_child_from_name(infdim_elt_n, 'ElementRange')
    infdim_elt_range = PT.get_value(infdim_elt_range_n)
    infdim_elt_offset = elt_range[1]-infdim_elt_range[0]+1
    infdim_elt_range[0] = infdim_elt_range[0]+infdim_elt_offset
    infdim_elt_range[1] = infdim_elt_range[1]+infdim_elt_offset
    PT.set_value(infdim_elt_range_n, infdim_elt_range)

    infdim_elt_name = PT.Element.CGNSName(infdim_elt_n)
    is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                PT.Subset.GridLocation(n)==CGNS_TO_LOC[infdim_elt_name]
    for elt_bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc):
      pl_n = PT.get_child_from_name(elt_bc_n, 'PointList')
      pl = PT.get_value(pl_n)
      PT.set_value(pl_n, pl+infdim_elt_offset)


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
  if PT.Zone.n_cell(zone)==n_elt:
    PT.set_value(zone, [[n_vtx+n_vtx_toadd, elt_range[1], 0]])
  else:
    PT.set_value(zone, [[n_vtx+n_vtx_toadd, PT.Zone.n_cell(zone), 0]])

  # # > Building old to new vtx num
  # old_vtx_num = np_utils.concatenate_np_arrays([vtx_match[0], new_vtx_num[0]])[1]
  # new_vtx_num = np_utils.concatenate_np_arrays([vtx_match[1], new_vtx_num[1]])[1]
  

  # > Report BCs from initial domain on periodic one
  old_elt_num = gc_elt_pl
  sort_old_elt_num = np.argsort(old_elt_num)
  new_elt_num = np.arange(n_elt, n_elt+n_elt_to_add, dtype=np.int32)

  # print(f'old_elt_num = {old_elt_num}')
  # print(f'new_elt_num = {new_elt_num}')

  is_asked_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_asked_bc):
    bc_name = PT.get_name(bc_n)
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    # print(f'    -> bc_pl {bc_pl-elt_offset}')
    # print(f'    -> gc_elt_pl {gc_elt_pl-1}')
    tag = np.isin(bc_pl-elt_offset, gc_elt_pl-1)
    new_bc_pl = bc_pl[tag]
    if new_bc_pl.size!=0:
      n_new_elt += new_bc_pl.size
      pl_idx = np.searchsorted(gc_elt_pl-1, new_bc_pl-elt_offset, sorter=sort_old_elt_num)
      # print(f'    -> bc to add {bc_name}: new_bc_pl.size = {new_bc_pl.size}')
      # print(f'       -> bc_pl     {bc_pl+elt_offset}')
      # print(f'       -> gc_elt_pl {gc_elt_pl}')
      # print(f'       -> new_bc_pl {new_bc_pl-elt_offset}')
      # print(f'       -> pl_idx    {pl_idx}')
      new_bc_pl = new_elt_num[sort_old_elt_num[pl_idx]]+elt_offset
      # print(f'       -> new_bc_pl {new_bc_pl}')

      new_bcs[bc_name+'p'] = new_bc_pl
      PT.new_BC(bc_name+'p',
                type='FamilySpecified',
                point_list=new_bc_pl.reshape((1,-1), order='F'),
                loc=CGNS_TO_LOC[cgns_name],
                family='PERIODIC',
                parent=zone_bc_n)

  # # We assume that elements are ordered by dim in CGNS (true for sonics)
  # # TODO : generalize ?
  # elt_vtx_pl = elmt_pl_to_vtx_pl(zone, gc_elt_pl, cgns_name)
  # print(f'elt_vtx_pl = {elt_vtx_pl}')
  # elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  # elt_n = elts_per_dim[3][0]
  # for dim in range(dim_elt-1,0,-1):
  #   assert len(elts_per_dim[dim])
  #   infdim_elt_n = elts_per_dim[dim][0]
  #   report_bcs_on_periodic_patch(zone,  infdim_elt_n, elt_vtx_pl, old_vtx_num, new_vtx_num, comm)
  

  return n_vtx_toadd, new_vtx_num, new_elt_pl


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
              type='FamilySpecified',
              point_list=np.arange(bar_elt_range[0]+n_bar, bar_elt_range[0]+n_bar+(n_vtx_toadd-1), dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)
  PT.new_BC(name='fixedp',
              type='FamilySpecified',
              point_list=np.arange(bar_elt_range[0]+n_bar+(n_vtx_toadd-1), bar_elt_range[0]+n_bar+(n_vtx_toadd-1)*2, dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)

  pl_fixed = new_num_vtx[0].reshape((1,-1), order='F')
  # pl_fixedp = np.arange(1, n_vtx_toadd+1, dtype=np.int32).reshape((1,-1), order='F') # not CGNS coherent, but tricks for periodic adaptation
  PT.new_BC(name='vtx_fixed',
            type='FamilySpecified',
            point_list=pl_fixed,
            loc='Vertex',
            family='BCS',
            parent=zone_bc_n)
  pl_fixedp = new_num_vtx[1].reshape((1,-1), order='F')
  # pl_fixedp = np.arange(1, n_vtx_toadd+1, dtype=np.int32).reshape((1,-1), order='F') # not CGNS coherent, but tricks for periodic adaptation
  PT.new_BC(name='vtx_fixedp',
            type='FamilySpecified',
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

  dist_base = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  is_3d = PT.get_value(dist_base)[0]==3

  zones = PT.get_nodes_from_label(dist_base, 'Zone_t')
  assert len(zones)==1
  zone = zones[0]
  n_tri_old = PT.Zone.n_cell(zone)

  # > Get GCs info
  zone_gc_n = PT.get_child_from_label(zone, 'ZoneGridConnectivity_t')
  gc_n = PT.get_child_from_name(zone_gc_n, gc_name)
  gc_vtx_n = PT.get_child_from_name(zone_gc_n, gc_name+'_0')
  gc_loc = PT.Subset.GridLocation(gc_n)
  periodic_values = PT.GridConnectivity.periodic_values(gc_n)

  gc_pl  = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_pld = PT.get_value(PT.get_child_from_name(gc_n, 'PointListDonor'))[0]
  gc_vtx_pl  = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointList'))[0]
  gc_vtx_pld = PT.get_value(PT.get_child_from_name(gc_vtx_n, 'PointListDonor'))[0]

  # > Add GCs as BCs
  PT.new_Family('PERIODIC', parent=dist_base)
  PT.new_Family('GCS', parent=dist_base)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  for gc_n in PT.get_nodes_from_label(zone_gc_n, 'GridConnectivity_t'):
    bc_name = PT.get_name(gc_n)
    bc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
    bc_loc  = PT.Subset.GridLocation(gc_n)
    PT.new_BC(name=bc_name,
              type='FamilySpecified',
              point_list=bc_pl,
              loc=bc_loc,
              family='GCS',
              parent=zone_bc_n)


  # > Add periodic elts connected to GC
  PT.print_tree(zone)
  # duplicate_vtx_pl = np.empty(0, dtype=np.int32)
  # new_num_vtx      = [np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)]


  # > Duplicate periodic cells:
  new_vtx_num = [np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)]
  for elt_name, loc in {'TETRA_4':'CellCenter', 'TRI_3':'FaceCenter', 'BAR_2':'EdgeCenter'}.items():
  # for elt_name, loc in {'TETRA_4':'CellCenter'}.items():
    print(f'elt_name = {elt_name} -> {gc_vtx_pld.size}')
    gc_elt_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, elt_name, elt_full=False)
    n_vtx_to_add, new_vtx_num, new_elt_pl = add_periodic_elmt(zone, gc_elt_pl,
                                                              gc_vtx_pl, gc_vtx_pld,
                                                              new_vtx_num,
                                                              periodic_values,
                                                              elt_name,
                                                              comm)


    # > Add volumic BCs so that we can delete patches after TODO: better way ?
    if elt_name=='TETRA_4':
      
      zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
      PT.new_BC(name=f'{elt_name.lower()}_periodic', 
                type='FamilySpecified',
                point_list=new_elt_pl.reshape((1,-1), order='F'),
                loc=loc,
                family='PERIODIC',
                parent=zone_bc_n)

      PT.new_BC(name=f'{elt_name.lower()}_constraint',   
                type='FamilySpecified',
                point_list=gc_elt_pl.reshape((1,-1), order='F'),
                loc=loc,
                family='PERIODIC',
                parent=zone_bc_n)

      # > Search undefined faces
      new_tri_pl = add_undefined_faces(zone, new_elt_pl, elt_name, gc_vtx_pl, 'TRI_3')
      PT.new_BC(name=f'tri_3_periodic', 
                type='FamilySpecified',
                point_list=new_tri_pl.reshape((1,-1), order='F'),
                loc='FaceCenter',
                family='PERIODIC',
                parent=zone_bc_n)

      new_tri_pl = add_undefined_faces(zone, gc_elt_pl, elt_name, gc_vtx_pld, 'TRI_3')
      PT.new_BC(name=f'tri_3_constraint', 
                type='FamilySpecified',
                point_list=new_tri_pl.reshape((1,-1), order='F'),
                loc='FaceCenter',
                family='PERIODIC',
                parent=zone_bc_n)

      n_cell = PT.Zone.n_cell(zone)
      fld = np.zeros(n_cell, dtype=np.float64)
      fld[new_elt_pl-1] = 2.
      fld[gc_elt_pl -1] = 1.
      PT.new_FlowSolution('FSolution', fields={'tag_face':fld}, loc='CellCenter', parent=zone)
      PT.rm_nodes_from_name(dist_tree, 'ZoneGridConnectivity')


      # > Search undefined faces




  # # > Create new BAR_2 elmts and associated BCs to constrain mesh adaptation
  # add_constraint_bcs(zone, new_num_vtx)

  

  return periodic_values, new_vtx_num


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


def add_undefined_faces(zone, elt_pl, elt_name, vtx_pl, tgt_elt_name):

  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==elt_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt      = PT.Element.Size(elt_n)
  dim_elt    = PT.Element.Dimension(elt_n)
  size_elt   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  n_elt_to_add = elt_pl.size
  # print(f'n_elt_to_add = {n_elt_to_add}')
  # print(f'elt_pl = {elt_pl}')
  # > Get elts connectivity
  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = elt_pl -1
  ec_pl  = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  ec_elt = ec[ec_pl]
  # print(f'tgt_elt_ec = {ec_elt.reshape(n_elt_to_add,size_elt)}')

  
  tag_elt = np.isin(ec_elt, vtx_pl, invert=True)
  # print(f'tag_elt = {tag_elt}')
  tag_elt_with_face = np.add.reduceat(tag_elt.astype(np.int32), np.arange(0,n_elt_to_add*size_elt,size_elt)) # True when has vtx 
  # print(f'tag_elt_with_face = {tag_elt_with_face}')
  elt_pl = elt_pl[np.where(tag_elt_with_face==size_elt-1)[0]]
  n_elt_to_add = elt_pl.size
  # print(f'PL_cell_with_face_to_add = {elt_pl}')

  # > Get elts connectivity
  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)
  pl     = elt_pl -1
  print(f'n_cell_with_face_to_add = {elt_pl.size}')
  ec_pl  = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  tgt_elt_ec = ec[ec_pl]

  

  conf0 = np.array([0, 1, 2], dtype=np.int32)
  # conf1 = np.array([0, 1, 3], dtype=np.int32)
  conf2 = np.array([0, 2, 3], dtype=np.int32)
  # conf3 = np.array([1, 2, 3], dtype=np.int32)
  
  conf0t = np.array([0, 2, 1], dtype=np.int32)
  # conf1t = np.array([0, 1, 3], dtype=np.int32)
  conf2t = np.array([0, 2, 1], dtype=np.int32)
  # conf3t = np.array([1, 2, 3], dtype=np.int32)

  tag_elt = np.isin(tgt_elt_ec, vtx_pl, invert=True)
  tag_elt_rshp = tag_elt.reshape(n_elt_to_add,size_elt)
  tag_eltm1 = np.where(tag_elt_rshp)
  tag_eltm1_rshp = tag_eltm1[1].reshape(n_elt_to_add,size_elt-1)

  tgt_elt_ec = tgt_elt_ec[tag_elt].reshape(n_elt_to_add,size_elt-1)
  for conf, conft in zip([conf0,conf2], [conf0t,conf2t]):
    tag_conf = np.where((tag_eltm1_rshp==conf).all(1))[0]
    tgt_elt_ec_cp = tgt_elt_ec[tag_conf]
    tgt_elt_ec_cp = tgt_elt_ec_cp[:,conft]
    tgt_elt_ec[tag_conf] = tgt_elt_ec_cp
  tgt_elt_ec = tgt_elt_ec.reshape(n_elt_to_add*(size_elt-1))


  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==tgt_elt_name
  tgt_elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  n_tgt_elt      = PT.Element.Size(tgt_elt_n)
  dim_tgt_elt    = PT.Element.Dimension(tgt_elt_n)
  size_tgt_elt   = PT.Element.NVtx(tgt_elt_n)
  tgt_elt_offset = PT.Element.Range(tgt_elt_n)[0]

  # Update target element
  tgt_ec_n   = PT.get_child_from_name(tgt_elt_n, 'ElementConnectivity')
  tgt_ec     = PT.get_value(tgt_ec_n)
  print(f'tgt_ec.size = {tgt_ec.size}')
  print(f'tgt_elt_ec.size = {tgt_elt_ec.size}')
  PT.set_value(tgt_ec_n, np.concatenate([tgt_ec, tgt_elt_ec]))

  tgt_er_n  = PT.get_child_from_name(tgt_elt_n, 'ElementRange')
  tgt_er    = PT.get_value(tgt_er_n)
  new_tgt_elt_pl = np.arange(tgt_er[1]+1, tgt_er[1]+1+n_elt_to_add)
  tgt_er[1] = tgt_er[1]+n_elt_to_add
  PT.set_value(tgt_er_n, tgt_er)

  update_infdim_elts(zone, dim_tgt_elt, tgt_er[1])


  # PT.new_BC(name=f'{elt_name.lower()}_periodic', 
  #           type='FamilySpecified',
  #           point_list=new_elt_pl.reshape((1,-1), order='F'),
  #           loc=loc,
  #           family='PERIODIC',
  #           parent=zone_bc_n)

  return new_tgt_elt_pl




def update_infdim_elts(zone, dim_elt, max_elt_range):

  # > Updating offset others elements
  elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  for dim in range(dim_elt-1,0,-1):
    assert len(elts_per_dim[dim])
    infdim_elt_n = elts_per_dim[dim][0]
    print(f'UPDATE {PT.get_name(infdim_elt_n)} elt')
    infdim_elt_range_n = PT.get_child_from_name(infdim_elt_n, 'ElementRange')
    infdim_elt_range = PT.get_value(infdim_elt_range_n)
    infdim_elt_offset = max_elt_range-infdim_elt_range[0]+1
    infdim_elt_range[0] = infdim_elt_range[0]+infdim_elt_offset
    infdim_elt_range[1] = infdim_elt_range[1]+infdim_elt_offset
    PT.set_value(infdim_elt_range_n, infdim_elt_range)

    infdim_elt_name = PT.Element.CGNSName(infdim_elt_n)
    is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                PT.Subset.GridLocation(n)==CGNS_TO_LOC[infdim_elt_name]
    for elt_bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc):
      pl_n = PT.get_child_from_name(elt_bc_n, 'PointList')
      pl = PT.get_value(pl_n)
      PT.set_value(pl_n, pl+infdim_elt_offset)
