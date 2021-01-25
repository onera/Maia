import Converter.Internal as I
import maia.sids.sids as sids
import numpy          as np
from maia.connectivity import connectivity_transform as CNT
from maia.connectivity.cgns_to_pdm_dmeshnodal import concatenate_bc

import Pypdm.Pypdm as PDM

# --------------------------------------------------------------------------

def cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone,comm):
  """
  """
  distrib_ud       = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
  distrib_vtx      = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell     = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  n_vtx = distrib_vtx[2]
  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]

  dn_cell = distrib_cell[1] - distrib_cell[0]

  elts = I.getNodesFromType1(dist_zone, 'Elements_t')
  n_sections = len(elts)
  n_elt3d = 0
  n_elt2d = 0
  n_elt1d = 0
  dn_elt3d = 0
  dn_elt2d = 0
  dn_elt1d = 0
  elt_connecs = []
  elt_pdm_types = np.zeros(n_sections, dtype=np.int32, order='F')
  elt_lengths =   np.zeros(n_sections, dtype=np.int32, order='F')
  for i,elt in enumerate(elts):
    elt_type = I.getValue(elt)[0]
    elt_type_name = sids.element_name(elt_type)
    if elt_type_name == "NGON_n":
      raise NotImplementedError # NGON
    elt_vtx = I.getNodeFromName1(elt, 'ElementConnectivity')[1]

    distrib_ngon_ud  = I.getNodeFromName1(elt           , ':CGNS#Distribution')
    distrib_elt     = I.getNodeFromName1(distrib_ngon_ud, 'Distribution'      )[1]

    n_elt_section = distrib_elt[2]
    dn_elt_section = distrib_elt[1] - distrib_elt[0]

    if elt_type_name in sids.element_types_of_dimension(3):
      n_elt3d += n_elt_section
      dn_elt3d += dn_elt_section
    if elt_type_name in sids.element_types_of_dimension(2):
      n_elt2d += n_elt_section
      dn_elt2d += dn_elt_section
    if elt_type_name in sids.element_types_of_dimension(1):
      n_elt1d += n_elt_section
      dn_elt1d += dn_elt_section

    elt_connecs += [elt_vtx]
    elt_pdm_types[i] = sids.cgns_elt_name_to_pdm_element_type(elt_type_name)
    elt_lengths[i] = dn_elt_section

  dmesh_nodal = PDM.DistributedMeshNodal(comm, n_vtx, n_elt3d, n_elt2d, n_elt1d)
  gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
  cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
  cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
  cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
  dvtx_coord = np.hstack(list(zip(cx, cy, cz)))
  dmesh_nodal.set_coordinnates(dvtx_coord)

  # keep dvtx_coord object alive for ParaDiGM
  multi_part_node = I.createUniqueChild(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  I.newDataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)

  dmesh_nodal.set_sections(elt_connecs,elt_pdm_types,elt_lengths)

  n_elmt_group, delmt_bound_idx, delmt_bound = concatenate_bc(dist_zone)
  dmesh_nodal.set_group_elmt(n_elmt_group, delmt_bound_idx, delmt_bound)

  return dmesh_nodal

## --------------------------------------------------------------------------
#def cgns_dist_tree_to_joinopp_array(dist_tree):
#  """
#  """
#  zones = I.getZones(dist_tree)
#
#  jns = []
#  for zone in zones:
#    # > Get ZoneGridConnectivity List
#    zone_gcs = I.getNodesFromType1(zone, 'ZoneGridConnectivity_t')
#    # > Get Join List if ZoneGridConnectivity is not None
#    #   - Match Structured and Match Hybride
#    if (zone_gcs != []):
#      jns += I.getNodesFromType1(zone_gcs, 'GridConnectivity_t')
#      jns += I.getNodesFromType1(zone_gcs, 'GridConnectivity1to1_t')
#  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  # > Count joins and declare array
#  join_to_opp = np.empty(len(jns), dtype='int32' )
#  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  # > Fill array
#  for jn in jns:
#    join_id     = I.getNodeFromName1(jn, 'Ordinal')[1]
#    join_opp_id = I.getNodeFromName1(jn, 'OrdinalJoinOpp')[1]
#    join_to_opp[join_id - 1] = join_opp_id - 1
#  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#  return join_to_opp
