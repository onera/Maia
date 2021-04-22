import Converter.Internal as I
import numpy          as np

import maia.sids.sids as sids
import maia.sids.Internal_ext as IE
from maia.utils import zone_elements_utils as EZU
from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.tree_exchange.dist_to_part.index_exchange import collect_distributed_pl

from Pypdm.Pypdm import DistributedMeshNodal

def cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone, comm, needs_vertex=True, needs_bc=True):
  """
  Create a pdm_dmesh_nodal structure from a distributed zone
  """
  distrib_vtx = IE.getDistribution(dist_zone, 'Vertex')
  n_vtx   = distrib_vtx[2]
  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]

  n_elt_per_dim  = [0,0,0]
  sorted_elts = EZU.get_ordered_elements_std(dist_zone)
  for elt in sorted_elts:
    elt_type = sids.ElementType(elt)
    assert sids.element_name(elt_type) != "NGON_n"
    n_elt_per_dim[sids.element_dim(elt_type)-1] += sids.ElementSize(elt)

  #Create DMeshNodal
  dmesh_nodal = DistributedMeshNodal(comm, n_vtx, *n_elt_per_dim[::-1])

  #Vertices
  if needs_vertex:
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
    I.newDataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)

  #Elements
  elt_pdm_types, elt_lengths = EZU.collect_pdm_type_and_nelemts(sorted_elts)
  elmts_connectivities = EZU.collect_connectity(sorted_elts)
  dmesh_nodal.set_sections(elmts_connectivities, elt_pdm_types, elt_lengths)

  # Boundaries
  if needs_bc:
    bc_point_lists = collect_distributed_pl(dist_zone, ['ZoneBC_t/BC_t'])
    delmt_bound_idx, delmt_bound = py_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)
    n_elmt_group = delmt_bound_idx.shape[0] - 1
    #Need an holder to prevent memory deletion
    pdm_node = I.createUniqueChild(dist_zone, ':CGNS#DMeshNodal#Bnd', 'UserDefinedData_t')
    I.newDataArray('delmt_bound_idx', delmt_bound_idx, parent=pdm_node)
    I.newDataArray('delmt_bound'    , delmt_bound    , parent=pdm_node)
    dmesh_nodal.set_group_elmt(n_elmt_group, delmt_bound_idx, delmt_bound)
  else:
    dmesh_nodal.set_group_elmt(0, np.zeros(1, dtype=np.int32), np.empty(0, dtype=pdm_gnum_dtype))

  return dmesh_nodal

