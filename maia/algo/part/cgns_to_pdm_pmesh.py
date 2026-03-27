import numpy as np
from mpi4py import MPI

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.utils                       import np_utils
from maia.factory.dist_from_part      import discover_nodes_from_matching, _recover_elements
from maia.pytree.maia import pdm_elts

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

def _add_sections_to_zone(zone_n:CGNSPartTree, sections:List, comm:MPIComm):
  elt_nodes      = PT.get_children_from_label(zone_n, 'Elements_t')
  last_elt_range = PT.Element.Range(elt_nodes[-1])[-1] if len(elt_nodes)!=0 else 0
  loc_offset     = 0

  g_shift_section = 0
  for i_section, section in enumerate(sections):
    n_elmt      = section["n_elmt"]
    np_numabs   = section["np_numabs"]

    l_shift_section = comm.allreduce(np_numabs.max(initial=0), MPI.MAX)

    if n_elmt == 0:
      g_shift_section += l_shift_section
      continue

    elt_range      = np.array([1, n_elmt], dtype=np.int32) + last_elt_range + loc_offset
    cgns_elmt_name = pdm_elts.pdm_elt_name_to_cgns_element_type(section["pdm_type"])
    elt_n          = PT.new_Elements(f"{cgns_elmt_name}.{i_section}",
                                     cgns_elmt_name,
                                     erange=elt_range,
                                     econn=section["np_connec"],
                                     parent=zone_n)

    gns = {"Element" :section["np_numabs"],
           "Sections":section["np_numabs"]+g_shift_section}
    if section['np_parent_entity_g_num'] is not None:
      gns["Entity"] = section['np_parent_entity_g_num']

    key = 'np_element_to_entity' if section['np_element_to_entity'] is not None else 'np_parent_num'
    if section[key] is not None:
      lnum_node = PT.new_node(':CGNS#LocalNumbering', 'UserDefinedData_t', parent=elt_n)
      PT.new_DataArray('Entity', section[key], parent=lnum_node)

    MT.new_GlobalNumbering(gns, parent=elt_n)
    loc_offset += n_elmt
    g_shift_section += l_shift_section

def build_cell_gnum(zone_n:CGNSPartTree) -> NDArray:
  dim = PT.Zone.CellDimension(zone_n)
  elts = PT.Zone.get_ordered_elements_per_dim(zone_n)[dim]
  elt_gnums = [PT.get_np_value(MT.get_GlobalNumbering(e, 'Sections')) for e in elts]
  idx, cat = np_utils.concatenate_np_arrays(elt_gnums, dtype=pdm_dtype)
  return cat

def _add_group(pdm_pmn, zone_n, i_part, pdm_geom_type):

  GEOM_TO_DIM = {PDM._PDM_GEOMETRY_KIND_VOLUMIC  : 3,
                 PDM._PDM_GEOMETRY_KIND_SURFACIC : 2,
                 PDM._PDM_GEOMETRY_KIND_RIDGE    : 1}
  GEOM_TO_LOC = {PDM._PDM_GEOMETRY_KIND_VOLUMIC  : 'CellCenter',
                 PDM._PDM_GEOMETRY_KIND_SURFACIC : 'FaceCenter',
                 PDM._PDM_GEOMETRY_KIND_RIDGE    : 'EdgeCenter'}
  GEOM_TO_NAME= {PDM._PDM_GEOMETRY_KIND_VOLUMIC  : 'cell_bc_',
                 PDM._PDM_GEOMETRY_KIND_SURFACIC : 'surf_bc_',
                 PDM._PDM_GEOMETRY_KIND_RIDGE    : 'line_bc_'}

  if (n_group := pdm_pmn.get_n_group(pdm_geom_type)) == 0:
    return

  zone_bc_n = PT.update_child(zone_n, "ZoneBC", "ZoneBC_t")
  dim_elt_range = PT.Zone.get_elt_range_per_dim(zone_n)
  elt_range = dim_elt_range[GEOM_TO_DIM[pdm_geom_type]]
  loc = GEOM_TO_LOC[pdm_geom_type]
  if loc == 'FaceCenter' and PT.Zone.CellDimension(zone_n) == 2:
    loc = 'CellCenter'

  for i_group in range(n_group):
    group_ids, group_gn = pdm_pmn.get_group(pdm_geom_type, i_part, i_group)
    bc_pl = group_ids + elt_range[0]-1
    bc_name = f"{GEOM_TO_NAME[pdm_geom_type]}{i_group}"
    if bc_pl.size!=0:
      bc_n = PT.new_BC(name=bc_name,
                       point_list=bc_pl.reshape((1,-1), order='F'),
                       loc=loc,
                       parent=zone_bc_n)
      MT.new_GlobalNumbering({'Index' : group_gn}, parent=bc_n)
      PT.new_node('Ordinal', 'Ordinal_t', i_group, parent=bc_n)


def part_zones_to_pdm_pmesh_nodal(part_zones: List[CGNSPartTree],
                                  comm: MPIComm,
                                  needs_bc:bool = False,
                                  igroup_from_ordinal:bool = False) -> PDM.PartMeshNodal:
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
    pdm_elt_type = pdm_elts.cgns_elt_name_to_pdm_element_type(PT.Element.Type(elmt))
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
          pmesh_nodal.group_set(pdm_elts.elements_dim_to_pdm_kind[i_dim+1], i_part, i_group, pl, gnum)

  return pmesh_nodal

def pdm_pmesh_nodal_to_part_zones(pdm_pmn:PDM.PartMeshNodal,
                                  comm:MPIComm,
                                  phy_dim:int = 3,
                                  zone_name:str = 'zone') -> List[CGNSPartTree]:
  part_zones = list()

  dim     = pdm_pmn.dim_get()
  n_part  = pdm_pmn.n_part_get()

  for i_part in range(n_part):

    sections_vol   = pdm_pmn.get_sections(PDM._PDM_GEOMETRY_KIND_VOLUMIC , i_part) if dim==3 else list()
    sections_surf  = pdm_pmn.get_sections(PDM._PDM_GEOMETRY_KIND_SURFACIC, i_part)
    sections_ridge = pdm_pmn.get_sections(PDM._PDM_GEOMETRY_KIND_RIDGE   , i_part)
    coordinates    = pdm_pmn.coord_get(i_part)
    sections_nat   = sections_surf if dim == 2 else sections_vol

    assert coordinates.size%3==0
    n_vtx  = coordinates.size//3
    n_cell = sum([section['n_elmt'] for section in sections_nat])

    zone_n = PT.new_Zone(name=MT.conv.add_part_suffix(zone_name, comm.rank, i_part),
                         type="Unstructured",
                         size=[[n_vtx, n_cell, 0]])
    coords = {f'Coordinate{d}' : coordinates[i::3] for i,d in enumerate('XYZ'[:phy_dim])}
    PT.new_GridCoordinates(fields=coords, parent=zone_n)

    _add_sections_to_zone(zone_n, sections_vol,   comm)
    _add_sections_to_zone(zone_n, sections_surf,  comm)
    _add_sections_to_zone(zone_n, sections_ridge, comm)

    if n_vtx==0 and n_cell==0:
      continue

    # > Create BCs
    if len(sections_surf) > 0:
      _add_group(pdm_pmn, zone_n, i_part, PDM._PDM_GEOMETRY_KIND_SURFACIC)
    if len(sections_ridge) > 0:
      _add_group(pdm_pmn, zone_n, i_part, PDM._PDM_GEOMETRY_KIND_RIDGE)

    MT.new_GlobalNumbering({'Vertex' : pdm_pmn.vtx_g_num_get(i_part),
                            'Cell'   : build_cell_gnum(zone_n)},
                            parent=zone_n)

    part_zones.append(zone_n)

  return part_zones