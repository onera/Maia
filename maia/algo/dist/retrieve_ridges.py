import Pypdm.Pypdm as PDM

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
from   maia                              import npy_pdm_gnum_dtype as pdm_dtype
from   maia.algo.dist.extract_part       import extract_elmt_connectivity_from_pl
from   maia.transfer                     import protocols as EP
from   maia.utils                        import np_utils, par_utils, as_pdm_gnum
from   maia.utils                        import logging as mlog
from   maia.typing                       import *

def replace_bc_identifiers(zone:CGNSTree, bc_identifiers:List[Union[str, List[str]]]) -> List[List[str]]:
  """
  For a given bc_identifiers, replace BC families with associated BC names from given zone. 
  """
  referenced_bcs          = list()
  replaced_bc_identifiers = list()
  
  for bc_identifier in bc_identifiers:

    # > Identify BCs
    if   isinstance(bc_identifier, str):
      identified_bcs = [PT.get_name(node) for node in PT.get_nodes_from_predicate(zone, lambda n : PT.get_label(n) == 'BC_t' and PT.predicate.belongs_to_family(n, bc_identifier))]
    elif isinstance(bc_identifier, list):
      identified_bcs = bc_identifier
    else:
      raise ValueError("bc_identifiers argument must be string or list.")

    # > Check if not empty
    if not identified_bcs:
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
  none_idx = np.array([i+1 for i in range(len(parents)) if parents[i] is     None], dtype=pdm_dtype)
  full_idx = np.array([i+1 for i in range(len(parents)) if parents[i] is not None], dtype=pdm_dtype)

  data_stri = np.array([len(parents[k-1]) for k in full_idx], np.int32)
  _, data   = np_utils.concatenate_np_arrays([parents[k-1] for k in full_idx], dtype=np.int32)

  out_stri, out = EP.part_to_part_strided([data_stri], [data], [full_idx], [none_idx], comm)

  # > Store parent groups info
  r_idx = 0
  for k,idx in enumerate(none_idx):
    size = out_stri[0][k]
    parents[idx-1] = out[0][r_idx:r_idx+size]
    r_idx += size

  return parents


def find_ridges(dist_tree: CGNSDistTree,
                bc_identifiers: Union[Literal['ALL_BCS'], List[Union[str, List[str]]]],
                comm: MPIComm) -> None:
  """Retrieve the edges delimiting specified BC surfaces of a volumic mesh.

  Tree is modified inplace: Elements_t nodes containing resulting edge elements
  are added in input tree.

  **Setting groups of surfaces**

  This function retrieves the edges that delimit groups of BCs, which have
  to be user-provided through the ``bc_identifiers`` list.
  Each group can be defined by either:

  - the name of each BCs belonging to the group *(list of str)*;
  - or a family name gathering the BCs *(str)*.

  Note that a given BC surface **must not** appear in more than one group.

  Note: 
    For convenience, the shortcut ``bc_identifiers='ALL_BCS'`` can be used to indicate that
    each BC constitutes an independant group.

  Args:
    dist_tree      (CGNSDistTree): Unstructured distributed tree, starting at Zone_t level or higher.
    bc_identifiers (list)        : List of BC groups bounded by searched edges (see above)
    comm           (MPIComm)     : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #retrieve_ridges@start
        :end-before: #retrieve_ridges@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  for zone in PT.iter_all_Zone_t(dist_tree):

    assert PT.Zone.CellDimension(zone)>1

    zone_dtype = PT.get_np_value(zone).dtype

    # > Transform bc_identifiers onto list of list of BCs
    if bc_identifiers == 'ALL_BCS':
      bc_identifiers = [[PT.get_name(bc)] for bc in PT.get_nodes_from_label(zone, 'BC_t')]
    replaced_bc_identifiers = replace_bc_identifiers(zone, bc_identifiers)

    # > Make unique PL for each group
    groups_cat_pl:List[NDArray] = []
    for bc_names in replaced_bc_identifiers:
      bcs = [PT.find_node_from_name_and_label(zone, bc_name, 'BC_t')  for bc_name in bc_names]
      group_pls = [PT.get_np_value(PT.find_child_from_name(bc, 'PointList')) for bc in bcs]
      groups_cat_pl.append(np_utils.concatenate_point_list(group_pls)[1])

    dgrp_face_idx, pl = np_utils.concatenate_np_arrays(groups_cat_pl)

    # > Get connectivity of surfacic elements and transfer it to pl "partition" for PDM
    elmt_2d_nodes = PT.Zone.get_ordered_elements_per_dim(zone)[2]
    face_vtx_idx, face_vtx = extract_elmt_connectivity_from_pl(zone, elmt_2d_nodes, pl, comm)

    # > Prepare arguments and call PDM function
    distrib_face      = par_utils.dn_to_distribution(pl.size, comm)
    distrib_face_full = par_utils.partial_to_full_distribution(distrib_face, comm)
    dgrp_face         = np.arange(distrib_face[0]+1, distrib_face[1]+1, dtype=pdm_dtype)

    distrib_ridge, dridge_vtx, \
    dgroup_edge_idx, dgroup_edge, \
    dridge_face_group_idx, dridge_face_group = PDM.dfind_topological_ridge(comm,
                                                                           distrib_face_full,  
                                                                           face_vtx_idx,
                                                                           as_pdm_gnum(face_vtx),
                                                                           dgrp_face_idx, 
                                                                           dgrp_face)
    if distrib_ridge[comm.size]==0:
      mlog.warning(f"no topological ridge found on given tree by find_ridges service.")
      return

    # > Create BAR elements in entry tree
    #   If already BAR node -> create new one and no check (but warning)
    #   We put BAR elts after already existing elts, because we can not safely place it before since we would need
    #     to renumber GC_t nodes, and this function can be called on a single zone
    # > Create element node name
    bar_nodes    = PT.get_children_from_predicate(zone, PT.predicate.is_elmt_of_type("BAR_2"))
    bar_names    = [PT.get_name(bar_n) for bar_n in bar_nodes]
    new_bar_name = 'topo_edge'
    i_name = 0
    while new_bar_name in bar_names:
      new_bar_name = f'topo_edge.{i_name}'
      i_name+=1

    # > Find where to insert 
    if len(PT.Zone.get_ordered_elements_per_dim(zone)[0])>0:
      raise NotImplementedError("Meshes with 0d elements aren't managed.")
    
    if PT.Zone.has_ngon_elements(zone) and not PT.Zone.has_nface_elements(zone):
      maia.algo.pe_to_nface(zone, comm)

    last_elt_n = PT.Zone.get_ordered_elements(zone)[-1]
    offset_new_bar = PT.Element.Range(last_elt_n)[1] + 1
      
    elt_range_edges = np.array([offset_new_bar,
                                offset_new_bar+distrib_ridge[-1]-1], dtype=pdm_dtype)
    elt_n = PT.new_Elements(new_bar_name, 'BAR_2',
                            erange=elt_range_edges.astype(zone_dtype, copy=False),
                            econn=dridge_vtx.astype(zone_dtype, copy=False),
                            parent=zone)
    dedges_partial_distrib = par_utils.full_to_partial_distribution(distrib_ridge, comm)
    MT.new_Distribution({'Element':dedges_partial_distrib}, parent=elt_n)


    # > Création des BCs EdgeCenter (une par face parent group) + descriptor qui stocke parent 1 et parent 2
    dgroup_edges = [dgroup_edge[dgroup_edge_idx[i]:dgroup_edge_idx[i+1]] for i in range(len(dgroup_edge_idx)-1)]
    parents = share_parent_bc_info(dedges_partial_distrib, dgroup_edges,
                                   dridge_face_group_idx, dridge_face_group, 
                                   comm)

    zbc_n = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
    for i, bc_edge in enumerate(parents):
      pl = dgroup_edges[i]+elt_range_edges[0]-1
      bc_egde_n = PT.new_BC(name=f'topo_ridge_{i+1}',
                            point_list=pl.reshape((1,-1), order='F').astype(zone_dtype, copy=False),
                            loc='EdgeCenter',
                            parent=zbc_n)
      MT.new_Distribution({'Index':par_utils.dn_to_distribution(pl.size, comm)}, parent=bc_egde_n)
      values = []
      for val in [bc_identifiers[k-1] for k in bc_edge]:
        if isinstance(val, str):
          values.append(f"{val}")
        else:
          values.append('[' + ', '.join(val) + ']')
      PT.new_Descriptor("Parents", '\n'.join(values), parent=bc_egde_n)
