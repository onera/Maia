import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils                       import np_utils
from maia.factory.dist_from_part      import discover_nodes_from_matching, _recover_elements
from maia.pytree.maia.pdm_elts        import cgns_elt_name_to_pdm_element_type, elements_dim_to_pdm_kind

import Pypdm.Pypdm as PDM

from maia.typing import *

def _soft_recover_elements(dist_zone:CGNSDistTree, part_zones:List[CGNSPartTree], comm:MPIComm):
  """
  Rebuild dist_zone elements from part_zones, with exact ElementRange but without rebuilding
  connectivity
  """
  # > Rename ElementConnectivity so _recover_elements won't rebuild connectivity (lighter process)
  for part_zone in part_zones:
    for elt_n in PT.get_children_from_label(part_zone, 'Elements_t'):
      elt_ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
      PT.set_name(elt_ec_n, '__ElementConnectivity__')

  _recover_elements(dist_zone, part_zones, comm)

  # > Retrieve ElementConnectivity initial name
  for part_zone in part_zones:
    for elt_n in PT.get_children_from_label(part_zone, 'Elements_t'):
      elt_ec_n = PT.get_child_from_name(elt_n, '__ElementConnectivity__')
      PT.set_name(elt_ec_n, 'ElementConnectivity')

def identify_n_group_bc(dist_zone:CGNSDistTree, loc:str) -> int:
  """
  Compute number of BCs of given grid_loc using ordinal or not
  """
  bcs = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.is_bc_of_location(loc)])
  bcs_ordinal_n = [PT.get_child_from_name(bc, 'Ordinal') for bc in bcs]
  
  if len(bcs) > 0 and all([ord is not None for ord in bcs_ordinal_n]):
    return max(PT.get_np_value(ord)[0] for ord in bcs_ordinal_n) + 1
  else:
    return len(bcs)


def cgns_part_zones_to_pdm_pmesh_nodal(part_zones: List[CGNSPartTree],
                                       comm: MPIComm,
                                       needs_bc:bool = False,
                                       igroup_from_ordinal:bool = False):
  """
  Create and return a pdm_pmesh_nodal structure from partitioned zones
  """

  # > Rebuild dist_tree structure from partitioned zones
  dist_zone = PT.new_Zone(type='Unstructured')
  _soft_recover_elements(dist_zone, part_zones, comm)
  if needs_bc:
    child_list = ['GridLocation_t'] + (['Ordinal_t'] if igroup_from_ordinal else [])
    discover_nodes_from_matching(dist_zone, part_zones, "ZoneBC_t/BC_t", comm,
                                 child_list=child_list, get_value='leaf')
    n_bcs = {loc: identify_n_group_bc(dist_zone, loc) for loc in ['EdgeCenter', 'FaceCenter', 'CellCenter']}
  else:
    n_bcs = {loc: 0 for loc in ['EdgeCenter', 'FaceCenter', 'CellCenter']}

  # > Create PartMeshNodal
  mesh_dim = PT.Zone.CellDimension(dist_zone)
  pmesh_nodal = PDM.PartMeshNodal(comm, len(part_zones), mesh_dim)

  # > Global settings (Elements)
  elmt_nodes = PT.Zone.get_ordered_elements(dist_zone)
  section_ids = list()
  for elmt in elmt_nodes:
    pdm_elt_type = cgns_elt_name_to_pdm_element_type(PT.Element.Type(elmt))
    id_section   = pmesh_nodal.add_section(pdm_elt_type)
    section_ids.append(id_section)

  # > Global settings (BCs)
  if needs_bc:

    if (n_bc_edge := n_bcs['EdgeCenter']) != 0:
      pmesh_nodal.n_group_set(PDM._PDM_GEOMETRY_KIND_RIDGE, n_bc_edge)
    if mesh_dim == 2:
      pmesh_nodal.n_group_set(PDM._PDM_GEOMETRY_KIND_SURFACIC, n_bcs['CellCenter'])
    else:
      if (n_bc_face := n_bcs['FaceCenter']) != 0:
        pmesh_nodal.n_group_set(PDM._PDM_GEOMETRY_KIND_SURFACIC, n_bc_face)
      pmesh_nodal.n_group_set(PDM._PDM_GEOMETRY_KIND_VOLUMIC, n_bcs['CellCenter'])

  # > Local settings
  for i_part, part_zone in enumerate(part_zones):

    # > Coordinates
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    if cz is None:
      cz = np.zeros_like(cx)
    pvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
    pvtx_ln_to_gn = MT.Zone.vtx_globalnumbering(part_zone)
    pmesh_nodal.set_coordinates(i_part, pvtx_coord, pvtx_ln_to_gn)

    # > Elements
    for id_section, elmt in zip(section_ids, elmt_nodes):

      if (pelmt := PT.get_child_from_name(part_zone, PT.get_name(elmt))) is None:
        continue

      n_elmt     = PT.Element.Size(pelmt)
      elmt_vtx   = PT.get_np_value(PT.find_child_from_name(pelmt, "ElementConnectivity"))
      elmt_g_num = PT.get_np_value(MT.find_GlobalNumbering(pelmt, 'Sections'))
      # We want the gnum in "all elements of same dim" numbering (not only current section)

      pmesh_nodal.set_section(id_section, i_part, elmt_vtx, elmt_g_num, None, None, n_elmt)

    # > BCs
    if needs_bc:

      if (part_zone_bc := PT.get_child_from_label(part_zone, 'ZoneBC_t')) is None:
        continue

      range_by_dim = PT.Zone.get_elt_range_per_dim(part_zone)

      # Skip Vertex-located BCs, because they are not in Element numbering
      elts_loc = ['EdgeCenter', 'CellCenter'] if mesh_dim == 2 else ['EdgeCenter', 'FaceCenter', 'CellCenter']
      for i_dim, elt_loc in enumerate(elts_loc):
        bcs = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.is_bc_of_location(elt_loc)])

        for i_group, dist_bc in enumerate(bcs):

          if (part_bc := PT.get_node_from_name(part_zone_bc, PT.get_name(dist_bc))) is None:
            continue

          if igroup_from_ordinal and (ordinal_n := PT.get_child_from_name(part_bc, 'Ordinal')) is not None:
            i_group = PT.get_np_value(ordinal_n)[0]

          pl_n = PT.find_child_from_name(part_bc, 'PointList')
          pl   = PT.get_value(pl_n)[0] - (range_by_dim[i_dim+1][0]-1)
          gnum = MT.Subset.globalnumbering(part_bc)
          pmesh_nodal.group_set(elements_dim_to_pdm_kind[i_dim+1], i_part, i_group, pl, gnum)

  return pmesh_nodal
