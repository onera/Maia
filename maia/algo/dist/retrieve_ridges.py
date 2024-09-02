import mpi4py.MPI as MPI

import Pypdm.Pypdm as PDM

import numpy as np

import maia
import maia.pytree as PT
from   maia                              import npy_pdm_gnum_dtype as pdm_dtype
from   maia.algo.apply_function_to_nodes import zones_iterator
from   maia.algo.part.point_cloud_utils  import create_sub_numbering
from   maia.utils                        import np_utils, par_utils, as_pdm_gnum


def replace_bc_identifiers(zone, bc_identifiers) -> list():
  """
  For a given bc_identifiers, replace BC families with associated BC names from given zone. 
  """
  referenced_bcs          = list()
  replaced_bc_identifiers = list()
  
  for bc_identifier in bc_identifiers:

    # > Identify BCs
    if   isinstance(bc_identifier, str):
      identified_bcs = [PT.get_name(node) for node in PT.get_nodes_from_predicate(zone, lambda n : PT.predicate.belongs_to_family(n, bc_identifier) and PT.get_label(n) == 'BC_t')]
    elif isinstance(bc_identifier, list):
      identified_bcs = bc_identifier
    else:
      raise ValueError("bc_identifiers argument must be string or list.")

    # > Check if not empty
    if identified_bcs == list():
      raise ValueError(f"Identifier \"{bc_identifier}\" match no result.")

    # > Replace while checking that it is not already defined
    for bc in identified_bcs:
      if bc in referenced_bcs:
        raise ValueError(f"BC \"{bc}\" identified from \"{bc_identifier}\" already referenced by another identifier.")
    replaced_bc_identifiers.append(identified_bcs)
    referenced_bcs         .extend(identified_bcs)

  return replaced_bc_identifiers


def share_parent_bc_info(dedge_distrib, dgroup_edges,
                         dridge_face_group_idx, dridge_face_group,
                         comm):
  '''
  Share dridge_face_group info to proc with missing edge group.
  '''
  # > Go through edge groups while saving parent groups
  parents = list()
  for dgroup_edge in dgroup_edges:
    if dgroup_edge.size == 0:
      parents.append(None)
    else:
      first_edge_idx = dgroup_edge[0]-1-dedge_distrib[0]
      parent_bcs = dridge_face_group[dridge_face_group_idx[first_edge_idx  ]:
                                     dridge_face_group_idx[first_edge_idx+1]]
      parents.append(np.sort(parent_bcs))

  # > Exchange parent groups info to procs where there are missing
  none_idx = np.array([i+1 for i in range(len(parents)) if parents[i] is     None], dtype=int)
  full_idx = np.array([i+1 for i in range(len(parents)) if parents[i] is not None], dtype=int)

  data_stri = np.array(  [len(parents[k-1]) for k in full_idx], np.int32)
  data      = np.concatenate([parents[k-1]  for k in full_idx])

  out_stri, out = maia.transfer.protocols.part_to_part_strided([data_stri], [data], [full_idx], [none_idx], comm)

  # > Store parent groups info
  r_idx = 0
  for k,idx in enumerate(none_idx):
    size = out_stri[0][k]
    parents[idx-1] = out[0][r_idx:r_idx+size]
    r_idx += size

  return parents


def find_boundary_edges(dist_tree, comm, bc_identifiers=list()) -> None:
  """Retrieve edges delimiting given BC surfaces of the input ``dist_tree``.

  Tree is modified in place: Elements nodes containing resulting lineic elements
  will be added to tree.

  **Setting groups of surface**

  Used BCs chosen for edge retrieving is available through the ``bc_identifiers`` argument,
  which must be a list, where each list element define a "group" of BCs. Resulting edges 
  will be the edges delimiting these "groups". They can be defined with:

  - *list* of all BC names belonging to a same "group"
  - *str* family name

  Warning:
    - If ``bc_identifiers`` list is empty, edges delimiting **all** bcs of ``dist_tree`` will be computed.
    - BCs of a different "groups" must not reference a same surfacic element.
    - If ``dist_tree`` isn't a Zone_t node, ``bc_identifiers`` argument will be used over all Zone_t nodes present in tree.

  Args:
    dist_tree      (CGNSTree): Unstructured tree
    comm           (MPIComm) : MPI communicator
    bc_identifiers (list, optional):
        List of BC Families or list of BCs from used to retrieve delimiting edges. Defaults to ``list()``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #retrieve_ridges@start
        :end-before: #retrieve_ridges@end
        :dedent: 2
  """
  
  new_edge_path = list()

  for zone in zones_iterator(dist_tree):

    assert PT.Zone.CellDimension(zone)>1

    # > Transform bc_identifiers onto list of list of BCs
    replaced_bc_identifiers = replace_bc_identifiers(zone, bc_identifiers)

    # > Make unique PL for each group
    bc_pls = []
    for bc_names in replaced_bc_identifiers:
      bcs = [PT.get_node_from_predicates(zone, f'ZoneBC_t/{bc_name}')  for bc_name in bc_names]
      bc_pls.append([PT.get_child_from_name(bc, 'PointList')[1] for bc in bcs])
    bc_pls = [np_utils.concatenate_point_list(pls)[1] for pls in bc_pls]
    dgrp_face_idx, pl = np_utils.concatenate_np_arrays(bc_pls)

    # > Get connectivity of surfacic elements and transfer it to pl "partition" for PDM
    is_2d_elmt = lambda n: PT.predicate.is_elmt_of_type(n, dim=2)
    face_vtx_strd, face_vtx = extract_elmt_connectivity_from_pl(zone, pl, comm,
                                                                elmt_predicate=is_2d_elmt)

    # > Prepare arguments and call PDM function
    distrib_face      = par_utils.dn_to_distribution(pl.size, comm)
    distrib_face_full = par_utils.partial_to_full_distribution(distrib_face, comm)
    dgrp_face         = np.arange(distrib_face[0]+1, distrib_face[1]+1, dtype=pdm_dtype)
    face_vtx_idx      = np_utils.sizes_to_indices(face_vtx_strd)

    distrib_ridge, dridge_vtx, \
    dgroup_edge_idx, dgroup_edge, \
    dridge_face_group_idx, dridge_face_group = PDM.dfind_topological_ridge(comm,
                                                                           distrib_face_full,  
                                                                           face_vtx_idx,
                                                                           as_pdm_gnum(face_vtx),
                                                                           dgrp_face_idx, 
                                                                           dgrp_face)

    # > Create BAR elements in entry tree
    #   If already BAR node -> create new one and no check (but warning)
    #   If ngon  entry -> add BAR after all other elements
    #   If nodal entry -> add BAR before or after previous elements (depending on entry order) -> may need to offset PL and element_range
    # > Create element node name
    is_bar_elmt  = lambda n: PT.predicate.is_elmt_of_type(n, cgns_name="BAR_2")
    bar_nodes    = PT.get_children_from_predicate(zone, is_bar_elmt)
    bar_names    = [PT.get_name(bar_n) for bar_n in bar_nodes]
    new_bar_name = 'topo_edge'
    i_name = 0
    while new_bar_name in bar_names:
      new_bar_name = f'topo_edge.{i_name}'
      i_name+=1

    # > Find where to insert 
    is_0d_elmt = lambda n: PT.predicate.is_elmt_of_type(n, dim=0)
    if len(PT.get_children_from_predicate(zone, is_0d_elmt))>0:
      raise NotImplementedError("Meshes with 0d elements aren't managed.")
    
    zone_ordering = PT.Zone.elt_ordering_by_dim(zone)
    offset_new_bar = 0
    if not PT.Zone.has_ngon_elements(zone) and zone_ordering==1:
      apply_offset_to_elts(zone, distrib_ridge[-1], 0)
    else:
      is_elmt = lambda n: PT.predicate.is_elmt_of_type(n)
      for elmt_n in PT.get_children_from_predicate(zone, is_2d_elmt):
        elmt_range = PT.Element.Range(elmt_n)
        offset_new_bar = max(offset_new_bar, elmt_range[1])
    offset_new_bar+=1
      
    elt_range_edges = np.array([offset_new_bar,
                                offset_new_bar+distrib_ridge[-1]], dtype=pdm_dtype)
    elt_n = PT.new_Elements(new_bar_name, 'BAR_2', erange=elt_range_edges, econn=dridge_vtx, parent=zone)
    dedges_partial_distrib = par_utils.full_to_partial_distribution(distrib_ridge, comm)
    PT.maia.new_distribution({'Element':dedges_partial_distrib}, parent=elt_n)

    # > Get path for new bar element node (may wont work if 2 zone has same name under 2 different base)
    is_current_zone  = lambda n: PT.get_label(n)=='Zone_t' and PT.get_name(n)==PT.get_name(zone)
    is_new_edge_elmt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2' and PT.get_name(n)==new_bar_name
    new_edge_path.append(PT.predicates_to_paths(dist_tree, ['CGNSBase_t', is_current_zone, is_new_edge_elmt])[0])


    # > Création des BCs EdgeCenter (une par face parent group) + descriptor qui stocke parent 1 et parent 2
    dgroup_edges = [dgroup_edge[dgroup_edge_idx[i]:dgroup_edge_idx[i+1]] for i in range(len(dgroup_edge_idx)-1)]
    parents = share_parent_bc_info(dedges_partial_distrib, dgroup_edges,
                                   dridge_face_group_idx, dridge_face_group, 
                                   comm)

    zbc_n = PT.get_node_from_label(zone, 'ZoneBC_t')
    if zbc_n is None:
      zbc_n = PT.new_ZoneBC(zone)
    for i, bc_edge in enumerate(parents):
      pl = dgroup_edges[i]+elt_range_edges[0]-1
      bc_egde_n = PT.new_BC(name=f'topo_ridge_{i+1}', point_list=pl.reshape((1,-1), order='F'), loc='EdgeCenter', parent=zbc_n)
      PT.maia.new_distribution({'Index':par_utils.dn_to_distribution(pl.size, comm)}, parent=bc_egde_n)
      values = []
      for val in [bc_identifiers[k-1] for k in bc_edge]:
        if isinstance(val, str):
          values.append(f"{val}")
        else:
          values.append('[' + ', '.join(val) + ']')
      PT.new_Descriptor("Parents", '\n'.join(values), parent=bc_egde_n)

  return new_edge_path


def extract_elmt_connectivity_from_pl(zone, pl, comm,
                                      elmt_predicate=lambda n: PT.predicate.is_elmt_of_type(n)):
  '''
  Return elmt connectivity of zone elements which are tagged in pl.
  '''
  delmt_conn_strd = list()
  delmt_conn      = list()
  delmt_gnum      = list()
  for elmt_n in PT.get_children_from_predicate(zone, elmt_predicate):
    elmt_distrib = PT.maia.get_distribution(elmt_n, 'Element')[1]
    elmt_range   = PT.Element.Range(elmt_n)
    elmt_conn    = PT.get_child_from_name(elmt_n, 'ElementConnectivity')[1]
    elmt_gnum    = np.arange(elmt_distrib[0], elmt_distrib[1], dtype=pdm_dtype) + elmt_range[0]
    if PT.Element.CGNSName(elmt_n) in ["NGON_n", "NFACE_n"]:
      elmt_conn_idx  = np_utils.safe_int_cast(PT.get_child_from_name(elmt_n, 'ElementStartOffset')[1], np.int32)
      elmt_conn_strd = np.diff(elmt_conn_idx)
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

  return elmt_conn_strd[0], elmt_conn[0]



def extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm,
                        bc_predicate=lambda n: PT.get_label(n)=='BC_t'):
  """
  Return distributed zone_bc node containing bc_predicate BCs from zone tagged.
  """
  # > Get predicate BC PLs
  bc_pls = list()
  for bc_n in PT.get_children_from_predicate(zone_bc_n, bc_predicate):
    pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pls.append(PT.get_value(pl_n)[0])

  # > Create part_to_part to identify intersecting BCs
  bc_ptp           = maia.transfer.protocols.PartToPart([pl], bc_pls, comm)
  ref_lnum2        = bc_ptp.get_referenced_lnum2()
  extract_new_gnum = np.arange(distri_pl[0], distri_pl[1], dtype=pdm_dtype)+1
  p2p_type         = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1
  req_id           = bc_ptp.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                  p2p_type,
                                  [extract_new_gnum])
  _, part2_data = bc_ptp.wait(req_id)

  # > Create intersecting BCs
  edge_zone_bc_n = PT.new_ZoneBC()
  for i_bc, bc_n in enumerate(PT.get_children_from_predicate(zone_bc_n, bc_predicate)):
    size_ref_lnum2_g = comm.allreduce(ref_lnum2[i_bc].size, op=MPI.SUM)
    if size_ref_lnum2_g!=0:
      bc_name = PT.get_name(bc_n)
      edge_bc_n = PT.new_BC(name=bc_name,
                            point_list=part2_data[i_bc].reshape((1,-1), order='F'),
                            parent=edge_zone_bc_n)
      bc_distri = par_utils.dn_to_distribution(part2_data[i_bc].size, comm)
      PT.maia.new_distribution({'Index':bc_distri}, parent=edge_bc_n)

  return edge_zone_bc_n


def extract_zone_edges(dist_zone, pl, comm): #-> CGNSTree:
  """
  Return distributed zone containing edges tagged in pl and associated BCs.
  """
  # > Extract edge_vtx from tagged edge in PL
  is_1d_elmt = lambda n: PT.predicate.is_elmt_of_type(n, dim=1)
  _, edge_vtx = extract_elmt_connectivity_from_pl(dist_zone, pl, comm,
                                                  elmt_predicate=is_1d_elmt)
  distri_edge = par_utils.dn_to_distribution(pl.size, comm)
  extract_edge_vtx = create_sub_numbering([edge_vtx], comm)
  
  # > Compute vtx pl from extracted edge_vtx
  vtx_distri = PT.maia.get_distribution(dist_zone, 'Vertex')[1]
  vtx_mask   = np.zeros(vtx_distri[1] - vtx_distri[0], bool)

  ptb = maia.transfer.protocols.PartToBlock(vtx_distri, [edge_vtx], comm)
  gnum = ptb.getBlockGnumCopy()
  vtx_mask[gnum-vtx_distri[0]-1] = True
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  extract_cx = cx[vtx_mask] ; extract_cy = cy[vtx_mask] ; extract_cz = cz[vtx_mask]
  distri_vtx = par_utils.dn_to_distribution(extract_cx.size, comm)

  # > Create edge zone node
  zone_name = PT.get_name(dist_zone)
  edge_zone = PT.new_Zone(zone_name, type="Unstructured",
                          size=[np.array([distri_vtx[-1], distri_edge[-1] , 0], dtype=pdm_dtype)])
  PT.maia.new_distribution({'Vertex':distri_vtx, 'Cell':distri_edge} , parent=edge_zone)
  
  PT.new_GridCoordinates('GridCoordinates', fields={'CoordinateX':extract_cx, 'CoordinateY':extract_cy, 'CoordinateZ':extract_cz}, parent=edge_zone)

  bar_name = "BAR_2"
  n_bar = edge_vtx.size/2
  distri_bar = par_utils.dn_to_distribution(n_bar, comm)

  extract_elmt_range = np.array([1, distri_bar[-1]], dtype=pdm_dtype)
  new_bar_n = PT.new_Elements(bar_name, 'BAR_2',
                              erange=extract_elmt_range,
                              econn=extract_edge_vtx,
                              parent=edge_zone)
  PT.maia.new_distribution({'Element':distri_bar}, parent=new_bar_n)

  # > Get BCs intersecting PL
  zone_bc_n      = PT.get_child_from_label(dist_zone, 'ZoneBC_t')
  is_edge_bc     = lambda n: PT.predicate.is_bc_of_loc(n, "EdgeCenter")
  edge_zone_bc_n = extract_bcs_from_pl(zone_bc_n, pl, distri_edge, comm, bc_predicate=is_edge_bc)
  for bc_n in PT.get_children_from_label(edge_zone_bc_n, 'BC_t'):
    PT.new_GridLocation(loc="CellCenter" , parent=bc_n)
  PT.add_child(edge_zone, edge_zone_bc_n)

  return edge_zone


def extract_edges(dist_tree, domain_pls, comm): #-> CGNSTree:
  """
  Extract edges defined by the provided PointList from a distributed tree.

  **Setting PointList by domains**

  Edge to extract can be controlled through the ``domain_pls`` argument,
  which must be :

  - *dict* of type ``{domain_path:point_list}`` if dist_tree is CGNSTree
  - point_list *array* if dist_tree is Zone_t

  Args:
    dist_tree      (CGNSTree)   : Unstructured CGNSTree or Zone_t
    domain_pls     (dict or str): PointList of edges to extract defined by domain
    comm           (MPIComm)    : MPI communicator

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
      zone_path = PT.utils.path_head(domain_path, 3)
      zone_n = PT.get_node_from_path(dist_tree, domain_path)
      edge_zone = extract_zone_edges(zone_n, domain_pl, comm)
      PT.add_child(edge_base, edge_zone)

  return edge_tree
