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
  elt_nodes = PT.get_nodes_from_predicate(zone, is_asked_elt)
  elt_conn  = [PT.get_value(PT.get_child_from_name(elt_n, 'ElementConnectivity')) for elt_n in elt_nodes]
  elt_conn  = np_utils.concatenate_np_arrays(elt_conn)[1]

  n_elt = PT.Element.Size(elt_nodes[0])
  size_elt = PT.Element.NVtx(elt_nodes[0])
  pl = pl - local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]]) -1
  conn_pl = np_utils.interweave_arrays([size_elt*pl+i_size for i_size in range(size_elt)])
  vtx_pl = np.unique(elt_conn[conn_pl])

  return vtx_pl


def tag_elmt_owning_vtx(n_elt, elmt_conn, pl):
  tag_vtx   = np.isin(elmt_conn, pl) # True where vtx is 
  tag_tri   = np.logical_or.reduceat(tag_vtx, np.arange(0,n_elt*3,3)) # True when has vtx 
  gc_tri_pl = np.where(tag_tri)[0]+1 # Which cells has vtx

  return gc_tri_pl


# > Add BCs on periodic patch
def add_bc_on_periodic_patch(zone, elt_pl, old_vtx_num, new_vtx_num, cgns_name, comm):
  print(f'\n\n > ADD BC ON PERIODIC PATCH {cgns_name}')
  elt_vtx_pl = elmt_pl_to_vtx_pl(zone, elt_pl, 'TRI_3')
  print(f'elt_vtx_pl = ({elt_vtx_pl.size}) {elt_vtx_pl}')

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
  print(f'invalid_elts_pl = ({ppatch_elts_pl.size})')

  is_asked_bc = lambda n: PT.get_label(n)=='BC_t' and\
                          PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  n_new_elt = 0
  new_bcs = dict()
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_asked_bc):
    bc_name = PT.get_name(bc_n)
    print(f'[{cgns_name}] BC {PT.get_value(bc_n)}')
    bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
    print(f'{ppatch_elts_pl} <- {bc_pl}')
    tag  = np.isin(bc_pl-offset, ppatch_elts_pl)
    print(f'{tag}')
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
  print(f'new_elt_pl = {new_elt_pl}')
  new_ec = -np.ones(n_new_elt*size_elt, dtype=np.int32)
  for isize in range(size_elt):
    new_ec[isize::size_elt] = ec[size_elt*(new_elt_pl-offset-1)+isize]
  print(f'new_ec = {new_ec}')

  n_vtx = PT.Zone.n_vtx(zone)
  old_to_new_vtx_num = -np.ones(n_vtx, dtype=np.int32)
  old_to_new_vtx_num[old_vtx_num-1] = new_vtx_num
  new_ec = old_to_new_vtx_num[new_ec-1]
  print(f'new_ec = {new_ec}')

  new_elt_num = np.arange(n_elt, n_elt+n_new_elt, dtype=np.int32)
  old_to_new_elt_num = -np.ones(n_elt, dtype=np.int32)
  old_to_new_elt_num[new_elt_pl-offset-1] = new_elt_num
  print(f'old_to_new_elt_num = {old_to_new_elt_num}')
  for bc_name in new_bcs.keys():
    print(f'bc_name {bc_name}')
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.print_tree(bc_n)
    pl_n = PT.get_child_from_name(bc_n, 'PointList')
    pl = PT.get_value(pl_n)[0]
    print(f'    pl = {pl}')
    new_pl = old_to_new_elt_num[pl-offset-1]+offset+1
    print(f'    new_pl = {new_pl}')
    PT.set_value(pl_n, new_pl.reshape((1,-1), order='F'))

  # > Update connectivity
  print(f'UPDATE CONNECTIVITY')
  PT.print_tree(elt_n)
  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)
  ec   = np.concatenate([ec, new_ec])
  PT.set_value(ec_n, ec)

  er_n  = PT.get_child_from_name(elt_n, 'ElementRange')
  er    = PT.get_value(er_n)
  er[1] = er[1] + n_new_elt
  PT.set_value(er_n, er)

  PT.print_tree(elt_n)


def add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, periodic_transfo, comm):
  elt_n = PT.get_node_from_name(zone, 'TRI_3.0')
  elt_conn_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  elt_conn    = PT.get_value(elt_conn_n)
  elt_size = 3 # TRI_3 for now

  print(f'N_GC_ELT = {gc_elt_pl.size}')
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
  print(f'vtx_transfo = {vtx_transfo}')
  print(f'new_num_vtx = {new_num_vtx}')

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

  # > Updating zone dimensions
  PT.set_value(zone, [[n_vtx+n_vtx_toadd, elt_range[1], 0]])
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  PT.add_child(base, zone)
  import maia
  maia.io.write_tree(tree, 'OUTPUT/square_extended_tmp.cgns')


  


  old_vtx_num = np_utils.concatenate_np_arrays([np.array(list(vtx_transfo.keys  ()),dtype=np.int32),
                                                np.array(list(new_num_vtx.keys  ()),dtype=np.int32)])[1]
  new_vtx_num = np_utils.concatenate_np_arrays([np.array(list(vtx_transfo.values()),dtype=np.int32),
                                                np.array(list(new_num_vtx.values()),dtype=np.int32)])[1]
  print(f'old_vtx_num = {old_vtx_num}')
  print(f'new_vtx_num = {new_vtx_num}')
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

  print(f'new_num_vtx = {new_num_vtx}')
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
  print(f'offset = {offset}')
  pl_c  = -np.ones(n_elt_to_rm*elt_size, dtype=np.int32)
  print(f'pl_c.shape  = {pl_c.shape}')
  print(f'pl          = {pl}')
  print(f'pl-offset-1 = {pl-offset-1}')
  for i_size in range(elt_size):
    pl_c[i_size::elt_size] = elt_size*(pl-offset-1)+i_size
  print(f'pl_c  = {pl_c}')
  print(f'ec.shape  = {ec.shape}')
  ec = np.delete(ec, pl_c)
  print(f'ec.shape  = {ec.shape}')

  PT.set_value(ec_n, ec)
  PT.set_value(er_n, er)

  # > Update BC PointList
  targets = -np.ones(pl.size, dtype=np.int32)
  elt_distri_ini = np.array([0,n_elt,n_elt], dtype=np.int32) # TODO pdm_gnum
  print(f'elt_distri_ini = {elt_distri_ini}')
  old_to_new_elt = merge_distributed_ids(elt_distri_ini, pl-offset, targets, comm, True)
  print(f'old_to_new_elt = {old_to_new_elt}')

  PT.print_tree(elt_n)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  print(f'[{cgns_name}] --> {CGNS_TO_LOC[cgns_name]}')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    print(f'[{cgns_name}] {PT.get_name(bc_n)}')
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl = PT.get_value(bc_pl_n)[0]
    print(f'          bc_pl = {bc_pl}')
    bc_pl = bc_pl-offset
    print(f'          bc_pl = {bc_pl}')
    in_pl = np.isin(bc_pl, pl)
    new_bc_pl = old_to_new_elt[bc_pl[np.invert(in_pl)]-1]
    tag_invalid_elt  = np.isin(new_bc_pl,-1)
    new_bc_pl = new_bc_pl[np.invert(tag_invalid_elt)]
    print(f'          new_bc_pl = {new_bc_pl}')
    new_bc_pl = new_bc_pl+offset
    print(f'          new_bc_pl = {new_bc_pl}')

    if new_bc_pl.size==0:
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
  PT.print_tree(elt_n)
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  print(f'[{cgns_name}] --> {CGNS_TO_LOC[cgns_name]}')
  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and\
                        PT.Subset.GridLocation(n)==CGNS_TO_LOC[cgns_name]
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    print(f'[{cgns_name}] {PT.get_name(bc_n)}')
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl = PT.get_value(bc_pl_n)[0]
    print(f'          bc_pl = {bc_pl}')
    new_bc_pl = bc_pl - offset
    print(f'          new_bc_pl = {new_bc_pl}')
    PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))


def find_invalid_elts(zone, cgns_name):

  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                           PT.Element.CGNSName(n)==cgns_name
  elt_n = PT.get_node_from_predicate(zone, is_asked_elt)
  ec_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec    = PT.get_value(ec_n)

  n_elt = PT.Element.Size(elt_n)
  size_elt = PT.Element.NVtx(elt_n)
  print(f'ec({cgns_name}) = {ec.shape} {ec}')
  tag_elt  = np.isin(ec,-1)
  tag_elt  = np.logical_or.reduceat(tag_elt, np.arange(0,n_elt*size_elt,size_elt)) # True when has vtx 
  invalid_elts_pl  = np.where(tag_elt)[0]+1
  print(f'tag_elt({cgns_name}) = {tag_elt.shape} {tag_elt}')

  offset = local_pl_offset(zone, LOC_TO_DIM[CGNS_TO_LOC[cgns_name]])

  return invalid_elts_pl+offset