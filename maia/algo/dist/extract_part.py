import Pypdm.Pypdm as PDM

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
from   maia                              import npy_pdm_gnum_dtype as pdm_dtype
from   maia.algo.part.point_cloud_utils  import create_sub_numbering
from   maia.transfer                     import protocols as EP
import maia.transfer.protocols
from   maia.utils                        import np_utils, par_utils


def extract_elmt_connectivity_from_pl(zone, elmt_nodes, pl, comm):
  '''
  Return elmt connectivity of zone elements which are tagged in pl.
  '''
  delmt_conn_strd = list()
  delmt_conn      = list()
  delmt_gnum      = list()
  for elmt_n in elmt_nodes:
    elmt_distrib = MT.distribution_value(elmt_n, 'Element')
    elmt_range   = PT.Element.Range(elmt_n)
    elmt_conn    = PT.get_child_from_name(elmt_n, 'ElementConnectivity')[1]
    elmt_gnum    = np.arange(elmt_distrib[0], elmt_distrib[1], dtype=pdm_dtype) + elmt_range[0]
    if PT.Element.CGNSName(elmt_n) in ["NGON_n", "NFACE_n"]:
      elmt_conn_idx  = PT.get_child_from_name(elmt_n, 'ElementStartOffset')[1]
      elmt_conn_strd = np.diff(elmt_conn_idx).astype(np.int32, copy=False)
    else:
      n_elmt         = elmt_distrib[1]-elmt_distrib[0]
      elmt_n_vtx     = PT.Element.NVtx(elmt_n)
      elmt_conn_strd = np.full(n_elmt, elmt_n_vtx, dtype=np.int32)
    
    delmt_conn_strd.append(elmt_conn_strd)
    delmt_conn     .append(elmt_conn)
    delmt_gnum     .append(elmt_gnum.astype(dtype=pdm_dtype))

  # > Create part_to_part to get connectivity in PL frame
  ptp       = maia.transfer.protocols.PartToPart(delmt_gnum, [pl], comm)
  ref_lnum2 = ptp.get_referenced_lnum2()[0]
  if ref_lnum2.size!=pl.size:
    raise RuntimeError("Elements referenced in pl missing in zone.")
  p2p_type  = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1
  req_id    = ptp.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                        p2p_type,
                        delmt_conn,
                        part1_stride=delmt_conn_strd)
  elmt_conn_strd, elmt_conn = ptp.wait(req_id)
  elmt_conn_idx = np_utils.sizes_to_indices(elmt_conn_strd[0])

  return elmt_conn_idx, elmt_conn[0]

def extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm,
                        bc_predicate=lambda n: PT.get_label(n)=='BC_t'):

  bc_pls = [PT.get_child_from_name(bc_n, 'PointList')[1][0] \
            for bc_n in PT.iter_children_from_predicate(zone_bc_n, bc_predicate)]
  
  extract_new_gnum = np.arange(distri_pl[0], distri_pl[1], dtype=pl.dtype) + 1
  intersections = EP.part_to_part([extract_new_gnum], [pl], bc_pls, comm)

  edge_zone_bc_n = PT.new_ZoneBC()
  for bc_n, intersection in zip(PT.get_children_from_predicate(zone_bc_n, bc_predicate), intersections):
    bc_distri = par_utils.dn_to_distribution(intersection.size, comm)
    if bc_distri[-1] > 0:
      bc_name = PT.get_name(bc_n)
      edge_bc_n = PT.new_BC(name=bc_name,
                            point_list=intersection.reshape((1,-1), order='F'),
                            parent=edge_zone_bc_n)
      PT.new_GridLocation(loc="CellCenter" , parent=edge_bc_n)
      MT.new_Distribution({'Index':bc_distri}, parent=edge_bc_n)

  return edge_zone_bc_n


def extract_zone_edges(dist_zone, pl, comm):
  """
  Return distributed zone containing edges tagged in pl and associated BCs.
  We assume that a same id does not appear twice in pl (?)
  """
  # > Extract edge_vtx from tagged edge in PL
  elmt_1d_nodes = PT.Zone.get_ordered_elements_per_dim(dist_zone)[1]
  _, edge_vtx = extract_elmt_connectivity_from_pl(dist_zone, elmt_1d_nodes, pl, comm)
  _extract_edge_vtx = create_sub_numbering([np_utils.safe_int_cast(edge_vtx, pdm_dtype)], comm)
  extract_edge_vtx = np_utils.safe_int_cast(_extract_edge_vtx[0], edge_vtx.dtype)
  
  distri_bar = par_utils.dn_to_distribution(extract_edge_vtx.size // 2, comm)

  # > Compute vtx pl from extracted edge_vtx
  vtx_distri = MT.distribution_value(dist_zone, 'Vertex')

  GI = EP.GlobalIndexer(vtx_distri, edge_vtx-1, comm)
  vtx_mask = (GI.access_counts > 0)
  coords =  PT.Zone.coordinates(dist_zone)._asdict()
  extract_coords = {key: coord[vtx_mask] for key,coord in coords.items()}
  distri_vtx = par_utils.dn_to_distribution(vtx_mask.sum(), comm)


  # > Create edge zone node
  edge_zone_size = np.array([[distri_vtx[-1], distri_bar[-1] , 0]], order='F', dtype=dist_zone[1].dtype)
  edge_zone = PT.new_Zone(PT.get_name(dist_zone), type="Unstructured", size=edge_zone_size)
  
  PT.new_GridCoordinates('GridCoordinates', fields=extract_coords, parent=edge_zone)

  extract_elmt_range = np.array([1, distri_bar[-1]], dtype=edge_zone_size.dtype)
  new_bar_n = PT.new_Elements('BAR_2', 'BAR_2',
                              erange=extract_elmt_range,
                              econn=extract_edge_vtx,
                              parent=edge_zone)
  MT.new_Distribution({'Element':distri_bar}, parent=new_bar_n)

  # > Get BCs intersecting PL
  zone_bc_n  = PT.get_child_from_label(dist_zone, 'ZoneBC_t')
  if zone_bc_n is not None:
    edge_zone_bc_n = extract_bcs_from_pl(zone_bc_n, pl, distri_bar, comm,
                      bc_predicate=PT.predicate.is_bc_of_loc('EdgeCenter'))
    PT.add_child(edge_zone, edge_zone_bc_n)

  MT.new_Distribution({'Vertex': distri_vtx, 'Cell': distri_bar} , parent=edge_zone)

  return edge_zone


def extract_edges(dist_tree, domain_pls, comm):
  """
  Extract edges defined by the provided PointList from a distributed tree.

  **Setting PointList by domains**

  Edge to extract can be controlled through the ``domain_pls`` argument,
  which must be :

  - *dict* of type ``{domain_path:point_list}`` if dist_tree is CGNSTree
  - point_list *array* if dist_tree is Zone_t

  Args:
    dist_tree      (CGNSDistTree) : Unstructured CGNSTree or Zone_t
    domain_pls     (dict or str)  : PointList of edges to extract defined by domain
    comm           (MPIComm)      : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #extract_edges@start
        :end-before: #extract_edges@end
        :dedent: 2
  """
  if PT.get_label(dist_tree)=='Zone_t':
    assert not isinstance(domain_pls, dict)
    edge_tree = extract_zone_edges(dist_tree, domain_pls, comm)
  else:
    assert isinstance(domain_pls, dict)
    edge_tree = PT.new_CGNSTree()
    for domain_path, domain_pl in domain_pls.items():
      base_name = domain_path.split('/')[0]
      edge_base = PT.update_child(edge_tree,
                                  name=base_name,
                                  label='CGNSBase_t',
                                  value=np.array([1,3], dtype=np.int32))  
      zone_n = PT.get_node_from_path(dist_tree, domain_path)
      edge_zone = extract_zone_edges(zone_n, domain_pl, comm)
      PT.add_child(edge_base, edge_zone)

  return edge_tree
