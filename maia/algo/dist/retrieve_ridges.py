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
  """Retrieve edges delimiting given (groups of) surfaces of the input ``dist_tree``.

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
    if PT.Zone.has_ngon_elements(zone):
      ngon_n = PT.Zone.NGonNode(zone)
      dface_vtx_idx  = np_utils.safe_int_cast(PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1], np.int32)
      dface_vtx      = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
      dface_distrib  = PT.maia.get_distribution(ngon_n, 'Element')[1]
      elt_range      = PT.get_child_from_name(ngon_n, 'ElementRange')[1]
      elt_range_min  = elt_range[0]
      dface_vtx_strd = np.diff(dface_vtx_idx)#[:dn_elmt]
      dn_elmt        = dface_distrib[1]-dface_distrib[0]
    else:
      raise NotImplementedError("U-Elements not implemented yet.")

    face_vtx_strd, face_vtx = maia.transfer.protocols.block_to_part_strided(dface_vtx_strd,
                                                                            dface_vtx,
                                                                            dface_distrib,
                                                                            [pl-elt_range_min+1],
                                                                            comm)


    # > Prepare arguments and call PDM function
    distrib_face      = par_utils.dn_to_distribution(pl.size, comm)
    distrib_face_full = par_utils.partial_to_full_distribution(distrib_face, comm)
    dgrp_face         = np.arange(distrib_face[0]+1, distrib_face[1]+1, dtype=pdm_dtype)
    face_vtx_idx      = np_utils.sizes_to_indices(face_vtx_strd[0])

    distrib_ridge, dridge_vtx, \
    dgroup_edge_idx, dgroup_edge, \
    dridge_face_group_idx, dridge_face_group = PDM.dfind_topological_ridge(comm,
                                                                           distrib_face_full,  
                                                                           face_vtx_idx,
                                                                           as_pdm_gnum(face_vtx[0]),
                                                                           dgrp_face_idx, 
                                                                           dgrp_face)

    # > Create BAR elements in entry tree
    #   If already BAR node -> create new one and no check (but warning)
    #   If ngon  entry -> add BAR after all other elements
    #   If nodal entry -> add BAR before or after previous elements (depending on entry order) -> may need to offset PL and element_range
    is_bar_elmt  = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2'
    bar_nodes    = PT.get_children_from_predicate(zone, is_bar_elmt)
    bar_names    = [PT.get_name(bar_n) for bar_n in bar_nodes]
    new_bar_name = 'BAR_2'
    i_name = 0
    while new_bar_name in bar_names:
      new_bar_name = f'BAR_2.{i_name}'
      i_name+=1

    if PT.Zone.has_ngon_elements(zone):
      elt_range_edges = np.array([elt_range[-1]+PT.Zone.n_cell(zone),
                                  elt_range[-1]+PT.Zone.n_cell(zone)+distrib_ridge[-1]], dtype=pdm_dtype)
      elt_n = PT.new_Elements(new_bar_name, 'BAR_2', erange=elt_range_edges, econn=dridge_vtx, parent=zone)
      dedges_partial_distrib = par_utils.full_to_partial_distribution(distrib_ridge, comm)
      PT.maia.new_distribution({'Element':dedges_partial_distrib}, parent=elt_n)
    else:
      raise NotImplementedError("U-Elements not implemented yet.")

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
      bc_egde_n = PT.new_BC(name=f'BCEdge_{i+1}', point_list=pl.reshape((1,-1), order='F'), loc='EdgeCenter', parent=zbc_n)
      PT.maia.new_distribution({'Index':par_utils.dn_to_distribution(pl.size, comm)}, parent=bc_egde_n)
      values = []
      for val in [bc_identifiers[k-1] for k in bc_edge]:
        if isinstance(val, str):
          values.append(f"{val}")
        else:
          values.append('[' + ', '.join(val) + ']')
      PT.new_Descriptor("Parents", '\n'.join(values), parent=bc_egde_n)

  return new_edge_path