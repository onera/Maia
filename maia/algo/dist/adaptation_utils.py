import maia.pytree        as PT
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
from   maia.utils         import np_utils

import numpy as np

def elmt_pl_to_vtx_pl(zone, pl):
  is_bar_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)=='BAR_2'
  bar_nodes = PT.get_nodes_from_predicate(zone, is_bar_elt)
  bar_con   = [PT.get_value(PT.get_child_from_name(bar_n, 'ElementConnectivity')) for bar_n in bar_nodes]
  bar_con   = np_utils.concatenate_np_arrays(bar_con)[1]

  pl = pl-local_pl_offset(zone, LOC_TO_DIM['EdgeCenter']) -1
  con_pl = np_utils.interweave_arrays([2*pl , 2*pl +1])

  vtx_pl = np.unique(bar_con[con_pl ])

  return vtx_pl


def tag_elmt_owning_vtx(n_elt, elmt_conn, pl):
  tag_vtx   = np.isin(elmt_conn, pl) # True where vtx is 
  tag_tri   = np.logical_or.reduceat(tag_vtx, np.arange(0,n_elt*3,3)) # True when has vtx 
  gc_tri_pl = np.where(tag_tri)[0] # Which cells has vtx

  return gc_tri_pl


def add_periodic_elmt(zone, gc_elt_pl, gc_vtx_pl, gc_vtx_pld, periodic_transfo):
  elt_n = PT.get_node_from_name(zone, 'TRI_3.0')
  elt_conn_n  = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  elt_conn    = PT.get_value(elt_conn_n)
  elt_size = 3 # TRI_3 for now

  n_elt_to_add = gc_elt_pl.size

  pelt_conn = -np.ones(n_elt_to_add*elt_size, dtype=np.int32)
  for isize in range(elt_size):
    pelt_conn[isize::3] = elt_conn[elt_size*gc_elt_pl+isize]

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

  # > Updating zone dimensions
  PT.set_value(zone, [[n_vtx+n_vtx_toadd, elt_range[1], 0]])

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
              point_list=np.arange(bar_elt_range[0]+n_bar, bar_elt_range[0]+n_bar+(n_vtx_toadd-1)*2, dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)