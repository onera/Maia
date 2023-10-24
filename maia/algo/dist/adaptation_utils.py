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
LOC_TO_CGNS = {'EdgeCenter':'BAR_2',
               'FaceCenter':'TRI_3',
               'CellCenter':'TETRA_4'}

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
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[elt_cgnsname]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_elt_bc):
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

  
  update_infdim_elts(zone, dim_elt, n_elt_to_add)


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

  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_elt_bc):
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

  return n_vtx_toadd, new_vtx_num, new_elt_pl, new_bcs.keys()


def detect_match_bcs(zone, match_elt_pl, gc_vtx_pl, gc_vtx_pld, cgns_name, comm):
  '''

  '''
  match_bcs = dict()
  if match_elt_pl.size==0:
    return match_bcs

  # > Get element infos
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n      = PT.get_node_from_predicate(zone, is_asked_elt)
  n_elt      = PT.Element.Size(elt_n)
  dim_elt    = PT.Element.Dimension(elt_n)
  size_elt   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  ec_n   = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec     = PT.get_value(ec_n)

  # > Go through all elt BCs, building connectivity of those fully included in GCdonor
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  
  vtx_gc_match      = [gc_vtx_pld, gc_vtx_pl]
  sort_vtx_gc_match = np.argsort(vtx_gc_match[0])

  # bc_to_match = dict()
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_elt_bc):
    bc_name = PT.get_name(bc_n)
    bc_pl   = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
    tag_elt_in_gc = np.isin(bc_pl-elt_offset, match_elt_pl-1)
    is_in_gc      = np.logical_and.reduce(tag_elt_in_gc)
    
    if is_in_gc:
      print(f'bc_name = {bc_name}')
      # > Get BC connectivity
      pl    = bc_pl - elt_offset
      ec_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
      ec_bc = ec[ec_pl]

      # > Creating connectivity like if duplicated
      idx_vtx_gc_match = np.searchsorted(vtx_gc_match[0], ec_bc, sorter=sort_vtx_gc_match)
      ec_bc = vtx_gc_match[1][sort_vtx_gc_match[idx_vtx_gc_match]]
      print(f'idx_vtx_match = {idx_vtx_gc_match}')
      
      # # > Save connectivity of bc
      # bc_to_match[bc_name] = ec_bc

      # > Search on other BCs if connectivity is the same
      for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_elt_bc):
        match_bc_name = PT.get_name(bc_n)
        print(f'  -> match_bc_name = {match_bc_name}')
        
        match_bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
        pl    = match_bc_pl - elt_offset
        ec_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
        ec_match_bc = ec[ec_pl]

        tag = np.isin(ec_match_bc, ec_bc)
        is_match = np.logical_and.reduce(tag)
        # print(f'     -> tag = {tag}')
        print(f'     -> is_match = {is_match}')
        if is_match:
          match_bcs[match_bc_name] = bc_name
          break

  return match_bcs



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
  print(f'\n\n\nREMOVING {pl.size} elts of {cgns_name}')

  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)
  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  offset = er[0]
  
  n_elt = er[1]-er[0]+1
  n_elt_to_rm = pl.size
  
  elt_dim  = PT.Element.Dimension(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  er[1] = er[1]-n_elt_to_rm
  last_elt = er[1]
  pl_c  = -np.ones(n_elt_to_rm*size_elt, dtype=np.int32)
  for i_size in range(size_elt):
    pl_c[i_size::size_elt] = size_elt*(pl-offset)+i_size
  ec = np.delete(ec, pl_c)

  PT.set_value(ec_n, ec)
  PT.set_value(er_n, er)

  # > Update BC PointList
  targets = -np.ones(pl.size, dtype=np.int32)
  elt_distri_ini = np.array([0,n_elt,n_elt], dtype=np.int32) # TODO pdm_gnum
  old_to_new_elt = merge_distributed_ids(elt_distri_ini, pl-offset+1, targets, comm, True)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl   = PT.get_value(bc_pl_n)[0]

    not_in_pl = np.isin(bc_pl, pl, invert=True)
    new_bc_pl = old_to_new_elt[bc_pl[not_in_pl]-offset]
    tag_invalid_elt = np.isin(new_bc_pl,-1, invert=True)
    new_bc_pl = new_bc_pl[tag_invalid_elt]
    new_bc_pl = new_bc_pl+offset-1

    if new_bc_pl.size==0:
      print(f'BC \"{PT.get_name(bc_n)}\" is empty, node is deleted.')
      PT.rm_child(zone_bc_n, bc_n)
    else:
      PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))

  update_infdim_elts(zone, elt_dim, -n_elt_to_rm)


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
  offset = PT.Element.Range(elt_n)[0]
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  tag_elt  = np.isin(ec,-1)
  tag_elt  = np.logical_or.reduceat(tag_elt, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  invalid_elts_pl  = np.where(tag_elt)[0]


  return invalid_elts_pl+offset


def merge_periodic_bc(zone, bc_names, vtx_tag, old_to_new_vtx_num, comm):
  n_vtx = PT.Zone.n_vtx(zone)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  # print(f'\nvtx_tag = {vtx_tag}')

  pbc1_n      = PT.get_child_from_name(zone_bc_n, bc_names[0])
  pbc1_loc    = PT.Subset.GridLocation(pbc1_n)
  pbc1_pl     = PT.get_value(PT.get_child_from_name(pbc1_n, 'PointList'))[0]
  pbc1_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc1_pl, LOC_TO_CGNS[pbc1_loc])
  # print(f'\npbc1_pl = {pbc1_pl}')
  # print(f'pbc1_vtx_pl = ({pbc1_vtx_pl.size}) {pbc1_vtx_pl}')

  pbc2_n      = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_loc    = PT.Subset.GridLocation(pbc2_n)
  pbc2_pl     = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  pbc2_vtx_pl = elmt_pl_to_vtx_pl(zone, pbc2_pl, LOC_TO_CGNS[pbc2_loc])
  # print(f'\npbc2_pl = {pbc2_pl}')
  # print(f'pbc2_vtx_pl = ({pbc2_vtx_pl.size}) {pbc2_vtx_pl}')


  old_vtx_num = np.flip(old_to_new_vtx_num[0]) # flip to debug, can be remove
  new_vtx_num = np.flip(old_to_new_vtx_num[1])
  # print(f'\nold_vtx_num = ({old_vtx_num.size}) {old_vtx_num}')
  # print(f'new_vtx_num = ({new_vtx_num.size}) {new_vtx_num}')
  # new_vtx_num = [array([10, 13, 19, 26, 29, 30, 31, 34, 35, 37, 38, 40, 42, 46, 47, 54, 64,
  #        65, 66, 67], dtype=int32), array([69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
  #        86, 87, 88])]


  # print(f'vtx_tag = {vtx_tag}')
  pl1_tag = vtx_tag[pbc1_vtx_pl-1]
  sort_old = np.argsort(old_vtx_num)
  idx_pl1_tag_in_old = np.searchsorted(old_vtx_num, pl1_tag, sorter=sort_old)
  # print(f'\npl1_tag = {pl1_tag}')
  # print(f'pl1_tag is in old_vtx_num = {np.isin(pl1_tag, old_vtx_num)}')
  # print(f'np.max(idx_pl1_tag_in_old) = {np.max(idx_pl1_tag_in_old)}')
  # print(f'old_vtx_num.size = {old_vtx_num.size}')


  pl2_tag = vtx_tag[pbc2_vtx_pl-1]
  sort_pl2_tag = np.argsort(pl2_tag)
  idx_new_in_pl2_tag = np.searchsorted(pl2_tag, new_vtx_num, sorter=sort_pl2_tag)
  # print(f'pl2_tag = {pl2_tag}')
  # print(f'pl2_tag is in old_vtx_num = {np.isin(pl2_tag, new_vtx_num)}')

  # print(f'test1 = {sort_old[idx_pl1_tag_in_old]}')
  # print(f'test2 = {idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]}')
  # print(f'test3 = {sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]}')

  sources = pbc2_vtx_pl[sort_pl2_tag[idx_new_in_pl2_tag[sort_old[idx_pl1_tag_in_old]]]]
  targets = pbc1_vtx_pl
  # print(f'\nsources = {sources}')
  # print(f'targets = {targets}')
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, sources, targets, comm, False)
  # print(f'\nold_to_new_vtx = {old_to_new_vtx}')


  remove_elts_from_pl(zone, pbc1_pl, LOC_TO_CGNS[pbc2_loc], comm)
  pbc2_n = PT.get_child_from_name(zone_bc_n, bc_names[1])
  pbc2_pl = PT.get_value(PT.get_child_from_name(pbc2_n, 'PointList'))[0]
  remove_elts_from_pl(zone, pbc2_pl, LOC_TO_CGNS[pbc2_loc], comm)
  update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, 'TETRA_4', comm)
  update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, 'TRI_3', comm)
  update_elt_vtx_numbering(zone, vtx_distri_ini, old_to_new_vtx, 'BAR_2', comm)

  n_vtx_to_rm = pbc2_vtx_pl.size
  # print(f'n_vtx_to_rm = {n_vtx_to_rm}')
  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  PT.set_value(cx_n, np.delete(cx, pbc2_vtx_pl-1))
  PT.set_value(cy_n, np.delete(cy, pbc2_vtx_pl-1))
  PT.set_value(cz_n, np.delete(cz, pbc2_vtx_pl-1))
  
  n_cell = PT.Zone.n_cell(zone)
  PT.set_value(zone, [[n_vtx-n_vtx_to_rm, n_cell, 0]])
  
  # return


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



  # PT.rm_child(zone_bc_n, pbc1_n)
  # PT.rm_child(zone_bc_n, pbc2_n)



def deplace_periodic_patch(zone, patch_name, gc_name, periodic_values,
                           bcs_to_update, bcs_to_retrieve, comm):

  # > Get_infos
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  patch_n       = PT.get_child_from_name(zone_bc_n, patch_name)
  patch_loc     = PT.Subset.GridLocation(patch_n)
  patch_pl      = PT.get_value(PT.get_child_from_name(patch_n, 'PointList'))[0]
  patch_vtx_pl  = elmt_pl_to_vtx_pl(zone, patch_pl, LOC_TO_CGNS[patch_loc])

  gc_n      = PT.get_child_from_name(zone_bc_n, gc_name)
  gc_loc    = PT.Subset.GridLocation(gc_n)
  gc_pl     = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_vtx_pl = elmt_pl_to_vtx_pl(zone, gc_pl, LOC_TO_CGNS[gc_loc])


  # > Duplicate GC
  print(f'bcs_to_retrieve = {bcs_to_retrieve}')
  n_new_vtx, pl_vtx_duplicate = duplicate_elts(zone, bcs_to_retrieve, comm)
  n_vtx = PT.Zone.n_vtx(zone)
  print(f'n_new_vtx = {n_new_vtx}')
  # sys.exit()

  # > Updating cell elements connectivity
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==LOC_TO_CGNS[patch_loc]
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_elt    = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)

  elt_gc_pl = tag_elmt_owning_vtx(zone, gc_vtx_pl, LOC_TO_CGNS[patch_loc], elt_full=False)
  select_elt = np.isin(elt_gc_pl, patch_pl)
  pl = elt_gc_pl[select_elt]-1
  n_elt_gc = pl.size
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  conn_cp = ec[conn_pl]
  mask    = np.isin(conn_cp, pl_vtx_duplicate)

  conn_pl = conn_pl[mask]
  conn_cp = conn_cp[mask]

  new_conn = np.searchsorted(pl_vtx_duplicate, conn_cp) #pl_vtx_duplicate already sorted by unique
  new_conn = new_conn+n_vtx+1
  ec[conn_pl] = new_conn
  PT.set_value(ec_n, ec)

  PT.set_value(zone, [[n_vtx+n_new_vtx, n_elt, 0]])


  # > Deplace vtx that are not on GC
  patch_vtx_pl   = elmt_pl_to_vtx_pl(zone, patch_pl, LOC_TO_CGNS[patch_loc]) #Connectivity has changed

  cx_n = PT.get_node_from_name(zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(zone)
  cx[patch_vtx_pl-1] = cx[patch_vtx_pl-1] + periodic_values[2][0] # Only translation managed for now TODO
  cy[patch_vtx_pl-1] = cy[patch_vtx_pl-1] + periodic_values[2][1] # Only translation managed for now TODO
  cz[patch_vtx_pl-1] = cz[patch_vtx_pl-1] + periodic_values[2][2] # Only translation managed for now TODO
  PT.set_value(cx_n, cx)
  PT.set_value(cy_n, cy)
  PT.set_value(cz_n, cz)


  # > Update BCs from periodic patch
  for elt_name, bc_names in bcs_to_update.items():
    is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                             PT.Element.CGNSName(n)==elt_name
    elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
    ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
    ec    = PT.get_value(ec_n)
    offset = PT.Element.Range(elt_n)[0]

    size_elt = PT.Element.NVtx(elt_n)

    is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          LOC_TO_CGNS[PT.Subset.GridLocation(n)]==elt_name
    zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
    bc_nodes  = PT.get_children_from_predicates(zone_bc_n, [is_elt_bc])
    for bc_n in bc_nodes:
      bc_name = PT.get_name(bc_n)
      if bc_name in bc_names:
        print(f'bc_name = {bc_name}')
        bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
        bc_pl = bc_pl - offset
        ec_pl = np_utils.interweave_arrays([size_elt*bc_pl+i_size for i_size in range(size_elt)])
        ec_bc = ec[ec_pl]
        mask  = np.isin(ec_bc, pl_vtx_duplicate)
        ec_pl = ec_pl[mask]
        ec_bc = ec_bc[mask]
        
        new_ec = np.searchsorted(pl_vtx_duplicate, ec_bc)
        new_ec = new_ec+n_vtx+1
        print(f'    --> ec[ec_pl] = {ec[ec_pl]}')
        ec[ec_pl] = new_ec
        print(f'    --> ec[ec_pl] = {ec[ec_pl]}')

        twin_bc_n    = PT.get_child_from_name(zone_bc_n, bc_name[:-1])
        twin_bc_pl_n = PT.get_child_from_name(twin_bc_n, 'PointList')
        twin_bc_pl = PT.get_value(twin_bc_pl_n)[0]
        bc_pl      = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
        PT.set_value(twin_bc_pl_n, np.concatenate([twin_bc_pl, bc_pl]).reshape((1,-1), order='F'))

        PT.rm_child(zone_bc_n, bc_n)


  # sys.exit()

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



  # PT.set_value(ec_n, ec)


def duplicate_elts(zone, bcs_to_duplicate, comm):
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')

  pl_vtx_duplicate = np.empty(0, dtype=np.int32)

  for elt_name, elt_bcs in bcs_to_duplicate.items():

    # > Get element 
    is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                             PT.Element.CGNSName(n)==elt_name
    elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
    n_elt      = PT.Element.Size(elt_n)
    dim_elt    = PT.Element.Dimension(elt_n)
    size_elt   = PT.Element.NVtx(elt_n)
    offset_elt = PT.Element.Range(elt_n)[0]

    ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
    ec    = PT.get_value(ec_n)

    for bc_name, new_bc_name in elt_bcs.items():
      bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
      bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]
      n_elt_to_add = bc_pl.size
      print(f'n_elt_to_add = {n_elt_to_add}')

      # print(f'elt_pl = {elt_pl}')
      # print(f'offset = {offset}')
      pl = bc_pl - offset_elt
      conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
      ec_duplicate = ec[conn_pl]


      # > Update connectivity with new vtx numbering
      n_vtx = PT.Zone.n_vtx(zone)
      l_pl_vtx_duplicate = np.unique(ec_duplicate)
      tag_vtx_to_create  = np.isin(l_pl_vtx_duplicate, pl_vtx_duplicate, invert=True)
      l_pl_vtx_duplicate = l_pl_vtx_duplicate[tag_vtx_to_create]
      pl_vtx_duplicate = np.concatenate([pl_vtx_duplicate, l_pl_vtx_duplicate])

      new_conn = np.searchsorted(pl_vtx_duplicate, ec_duplicate) #pl_vtx_duplicate already sorted by unique
      new_conn = new_conn +n_vtx+1

      # if elt_name=='TRI_3':
      #   # print(f'new_conn = {new_conn}')
      #   new_conn = new_conn.reshape(n_elt_to_add,size_elt)
      #   rotation = np.array([0,2,1], dtype=np.int32)
      #   new_conn = new_conn[:,rotation]
      #   new_conn = new_conn.reshape(n_elt_to_add*size_elt)
      #   # print(f'new_conn = {new_conn}')
      #   # sys.exit()

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

      new_bc_pl = np.arange(er[1]-n_elt_to_add+1, er[1]+1, dtype=np.int32)
  
      PT.new_BC(new_bc_name,
                type='FamilySpecified',
                point_list=new_bc_pl.reshape((1,-1), order='F'),
                loc=CGNS_TO_LOC[elt_name],
                family='BCS',
                parent=zone_bc_n)
  
      update_infdim_elts(zone, dim_elt, n_elt_to_add)


  n_vtx_duplicate = pl_vtx_duplicate.size
  return n_vtx_duplicate, pl_vtx_duplicate



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

  # > rm GC vtx nodes, which are new useless
  is_vtx_gc = lambda n: PT.get_label(n)=='GridConnectivity_t' and PT.Subset.GridLocation(n)=='Vertex'
  PT.rm_nodes_from_predicate(dist_tree, is_vtx_gc)


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
  # PT.print_tree(zone)
  # duplicate_vtx_pl = np.empty(0, dtype=np.int32)
  # new_num_vtx      = [np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)]


  # > Duplicate periodic cells:
  periodized_bcs = dict()
  matching_bcs   = dict()
  new_vtx_num = [np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)]
  for elt_name, loc in {'TETRA_4':'CellCenter', 'TRI_3':'FaceCenter', 'BAR_2':'EdgeCenter'}.items():
  # for elt_name, loc in {'TETRA_4':'CellCenter'}.items():
    print(f'elt_name = {elt_name} -> {gc_vtx_pld.size}')
    gc_elt_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, elt_name, elt_full=False)
    # print(f'  --> n_elt_tagged = {gc_elt_pl.size}')
    # if elt_name=='TRI_3':
    #   gc_elt_pl = tag_elmt_owning_vtx(zone, np.concatenate([gc_vtx_pld, new_vtx_num[0]]), elt_name, elt_full=True)
    # print(f'  --> n_elt_tagged = {gc_elt_pl.size}')
    n_vtx_to_add, new_vtx_num, new_elt_pl, periodized_bcs[elt_name] =\
      add_periodic_elmt(zone, gc_elt_pl,
                        gc_vtx_pl, gc_vtx_pld,
                        new_vtx_num,
                        periodic_values,
                        elt_name,
                        comm)
    match_elt_pl = tag_elmt_owning_vtx(zone, gc_vtx_pld, elt_name, elt_full=True)
    matching_bcs[elt_name] = detect_match_bcs(zone, match_elt_pl, gc_vtx_pl, gc_vtx_pld, elt_name, comm)

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

  pl_constraint = new_vtx_num[0].reshape((1,-1), order='F')
  PT.new_BC(name='vtx_constraint',
            type='FamilySpecified',
            point_list=pl_constraint,
            loc='Vertex',
            family='PERIODIC',
            parent=zone_bc_n)
  pl_periodic = new_vtx_num[1].reshape((1,-1), order='F')
  PT.new_BC(name='vtx_periodic',
            type='FamilySpecified',
            point_list=pl_periodic,
            loc='Vertex',
            family='PERIODIC',
            parent=zone_bc_n)

  return periodic_values, new_vtx_num, periodized_bcs, matching_bcs


def retrieve_initial_domain(dist_tree, gc_name, periodic_values, new_vtx_num,
                            bcs_to_update, bcs_to_retrieve, comm):


  dist_base = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  is_3d = PT.get_value(dist_base)[0]==3

  dist_zone = PT.get_child_from_label(dist_base, 'Zone_t')

  n_vtx = PT.Zone.n_vtx(dist_zone)

  zone_bc_n = PT.get_child_from_label(dist_zone, 'ZoneBC_t')
  # PT.print_tree(zone_bc_n)

  cell_elt_name = 'TETRA_4' if is_3d else 'TRI_3'
  face_elt_name = 'TRI_3'   if is_3d else 'BAR_2'
  edge_elt_name = 'BAR_2'   if is_3d else  None

  # > Removing old periodic patch
  bc_to_rm_n      = PT.get_child_from_name(zone_bc_n, f'{cell_elt_name.lower()}_constraint')
  bc_to_rm_loc    = PT.Subset.GridLocation(bc_to_rm_n)
  bc_to_rm_pl     = PT.get_value(PT.get_child_from_name(bc_to_rm_n, 'PointList'))[0]
  bc_to_rm_vtx_pl = elmt_pl_to_vtx_pl(dist_zone, bc_to_rm_pl, cell_elt_name)
  print(f'n_vtx_to_rm = {bc_to_rm_vtx_pl.size}')
  n_elt_to_rm     = bc_to_rm_pl.size
  print(f'n_elt_to_rm = {n_elt_to_rm}')

  bc_to_keep_n      = PT.get_child_from_name(zone_bc_n, f'{face_elt_name.lower()}_constraint')
  bc_to_keep_loc    = PT.Subset.GridLocation(bc_to_keep_n)
  bc_to_keep_pl     = PT.get_value(PT.get_child_from_name(bc_to_keep_n, 'PointList'))[0]
  bc_to_keep_vtx_pl = elmt_pl_to_vtx_pl(dist_zone, bc_to_keep_pl, face_elt_name)
  print(f'n_vtx_to_keep = {bc_to_keep_vtx_pl.size}')
  n_vtx_to_keep     = bc_to_keep_vtx_pl.size


  tag_vtx = np.isin(bc_to_rm_vtx_pl, bc_to_keep_vtx_pl, invert=True) # True where vtx is 
  # preserved_vtx_id = bc_to_rm_vtx_pl[tag_vtx][0]
  bc_to_rm_vtx_pl = bc_to_rm_vtx_pl[tag_vtx]
  n_vtx_to_rm = bc_to_rm_vtx_pl.size
  print(f'n_vtx_to_rm = {n_vtx_to_rm}')

  # > Compute new vtx numbering
  vtx_tag_n = PT.get_node_from_name(dist_zone, 'vtx_tag')
  vtx_tag   = PT.get_value(vtx_tag_n)
  vtx_tag = np.delete(vtx_tag, bc_to_rm_vtx_pl-1)
  PT.set_value(vtx_tag_n, vtx_tag)

  ids = bc_to_rm_vtx_pl
  targets = -np.ones(bc_to_rm_vtx_pl.size, dtype=np.int32)
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, ids, targets, comm, True)

  cx_n = PT.get_node_from_name(dist_zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(dist_zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(dist_zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  PT.set_value(cx_n, np.delete(cx, bc_to_rm_vtx_pl-1))
  PT.set_value(cy_n, np.delete(cy, bc_to_rm_vtx_pl-1))
  PT.set_value(cz_n, np.delete(cz, bc_to_rm_vtx_pl-1))


  update_elt_vtx_numbering(dist_zone, vtx_distri_ini, old_to_new_vtx, cell_elt_name, comm)
  n_cell = remove_elts_from_pl(dist_zone, bc_to_rm_pl, cell_elt_name, comm)

  update_elt_vtx_numbering(dist_zone, vtx_distri_ini, old_to_new_vtx, 'TRI_3', comm)
  invalid_elt_pl = find_invalid_elts(dist_zone, 'TRI_3')
  n_tri = remove_elts_from_pl(dist_zone, invalid_elt_pl, 'TRI_3', comm)

  update_elt_vtx_numbering(dist_zone, vtx_distri_ini, old_to_new_vtx, 'BAR_2', comm)
  invalid_elt_pl = find_invalid_elts(dist_zone, 'BAR_2')
  n_bar = remove_elts_from_pl(dist_zone, invalid_elt_pl, 'BAR_2', comm)
  
  PT.set_value(dist_zone, [[n_vtx-n_vtx_to_rm, n_cell, 0]])
  
  

  bc_names = list()
  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='EdgeCenter'
  zone_bc_n = PT.get_node_from_label(dist_tree, 'ZoneBC_t')
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_edge_bc):
    bc_names.append(PT.get_name(bc_n))
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'FaceCenter')
  
  maia.io.write_tree(dist_tree, 'OUTPUT/constraint_rmvd.cgns')

  for bc_name in bc_names:
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'EdgeCenter')


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


  # > Deplace periodic patch to retrieve initial domain
  print(f'bcs_to_update = {bcs_to_update}')
  deplace_periodic_patch(dist_zone, f'{cell_elt_name.lower()}_periodic',
                         gc_name, periodic_values,
                         bcs_to_update, bcs_to_retrieve, comm)
  # maia.io.write_tree(dist_tree, 'OUTPUT/ppatch_deplaced.cgns')
  vtx_tag_n = PT.get_node_from_name(dist_zone, 'vtx_tag')
  vtx_tag   = PT.get_value(vtx_tag_n)
  merge_periodic_bc(dist_zone,
                    [f'{face_elt_name.lower()}_constraint', f'{face_elt_name.lower()}_periodic'],
                    vtx_tag,
                    new_vtx_num,
                    comm)
  # maia.io.write_tree(dist_tree, 'OUTPUT/ppatch_merged.cgns')


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

  update_infdim_elts(zone, dim_tgt_elt, n_elt_to_add)


  # PT.new_BC(name=f'{elt_name.lower()}_periodic', 
  #           type='FamilySpecified',
  #           point_list=new_elt_pl.reshape((1,-1), order='F'),
  #           loc=loc,
  #           family='PERIODIC',
  #           parent=zone_bc_n)

  return new_tgt_elt_pl




def update_infdim_elts(zone, dim_elt, offset):

  # > Updating offset others elements
  elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  for dim in range(dim_elt-1,0,-1):
    assert len(elts_per_dim[dim])
    infdim_elt_n = elts_per_dim[dim][0]
    print(f'UPDATING {PT.get_name(infdim_elt_n)} with offset = {offset}')

    infdim_elt_range_n = PT.get_child_from_name(infdim_elt_n, 'ElementRange')
    infdim_elt_range = PT.get_value(infdim_elt_range_n)
    print(f'   --> elt_range = {infdim_elt_range}')
    infdim_elt_range[0] = infdim_elt_range[0]+offset
    infdim_elt_range[1] = infdim_elt_range[1]+offset
    print(f'   --> new elt_range = {infdim_elt_range}')
    PT.set_value(infdim_elt_range_n, infdim_elt_range)

    infdim_elt_name = PT.Element.CGNSName(infdim_elt_n)
    is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                PT.Subset.GridLocation(n)==CGNS_TO_LOC[infdim_elt_name]
    for elt_bc_n in PT.get_nodes_from_predicate(zone, is_elt_bc):
      pl_n = PT.get_child_from_name(elt_bc_n, 'PointList')
      pl = PT.get_value(pl_n)
      PT.set_value(pl_n, pl+offset)
