import Converter.Internal as I
import numpy          as np

import maia.sids.sids as sids
from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
#from maia.tree_exchange.dist_to_part.index_exchange import collect_distributed_pl

from Pypdm.Pypdm import DistributedMeshNodal

def cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone,comm):
  """
  Create a pdm_dmesh_nodal structure from a distributed zone
  """
  distrib_ud       = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
  distrib_vtx      = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell     = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  n_vtx   = distrib_vtx[2]
  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  dn_cell = distrib_cell[1] - distrib_cell[0]

  elts = I.getNodesFromType1(dist_zone, 'Elements_t')
  n_elt_per_dim  = [0,0,0]
  dn_elt_per_dim = [0,0,0]
  elt_vtx_list    = []
  elt_pdm_types = np.zeros(len(elts), dtype=np.int32)
  elt_lengths =   np.zeros(len(elts), dtype=np.int32)

  for i,elt in enumerate(elts):
    elt_type      = sids.ElementType(elt)
    elt_type_name = sids.element_name(elt_type)
    if elt_type_name == "NGON_n":
      raise NotImplementedError # NGON

    elt_vtx        = I.getNodeFromName1(elt           , 'ElementConnectivity')[1]
    distrib_elt_ud = I.getNodeFromName1(elt           , ':CGNS#Distribution')
    distrib_elt    = I.getNodeFromName1(distrib_elt_ud, 'Element'      )[1]

    n_elt_section  = distrib_elt[2]
    dn_elt_section = distrib_elt[1] - distrib_elt[0]

    for idim in range(3):
      if elt_type_name in sids.element_types_of_dimension(idim+1):
        n_elt_per_dim [idim] += n_elt_section
        dn_elt_per_dim[idim] += dn_elt_section
        break #Element can not have 2 dims

    elt_vtx_list.append(elt_vtx)
    elt_pdm_types[i] = sids.cgns_elt_name_to_pdm_element_type(elt_type_name)
    elt_lengths[i] = dn_elt_section

  dmesh_nodal = DistributedMeshNodal(comm, n_vtx, *n_elt_per_dim[::-1])
  if dn_vtx > 0:
    gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
    cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
    cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
    dvtx_coord = py_utils.interweave_arrays([cx,cy,cz])
  else:
    dvtx_coord = np.empty(0, dtype='float64', order='F')
  dmesh_nodal.set_coordinnates(dvtx_coord)

  # keep dvtx_coord object alive for ParaDiGM
  multi_part_node = I.createUniqueChild(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  I.newDataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)

  dmesh_nodal.set_sections(elt_vtx_list, elt_pdm_types, elt_lengths)

  # Boundaries
  # bc_point_lists = collect_distributed_pl(dist_zone, ['ZoneBC_t/BC_t'])
  # delmt_bound_idx, delmt_bound = py_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)
  # n_elmt_group = delmt_bound_idx.shape[0] - 1
  # dmesh_nodal.set_group_elmt(n_elmt_group, delmt_bound_idx, delmt_bound)
  dmesh_nodal.set_group_elmt(0, np.zeros(1, dtype=np.int32), np.empty(0, dtype=pdm_gnum_dtype))

  return dmesh_nodal

