import Converter.Internal as I

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import maia.sids.sids as SIDS
from maia.utils import py_utils
from maia.utils import zone_elements_utils as EZU
from maia.tree_exchange.dist_to_part.index_exchange import collect_distributed_pl

import Pypdm.Pypdm as PDM
from maia import npy_pdm_gnum_dtype as npy_pdm_gnum_t

def cgns_to_pdm(zone, comm):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  dmesh_nodal = PDM.DistributedMeshNodal(comm, n_vtx, n_cell)

  # dvtx_coord  = np.empty(n_vtx*3, dtype='double', order='C')
  # dmesh_nodal.set_coordinnates(dvtx_coord)
  # print("TODO REMOVE COORDINATE in generate_ngon_from_std_elements.pyx")

  sorted_elmts = EZU.get_ordered_elements_std(zone)

  # nb_elemts = sum([SIDS.ElementSize(elmt) for elmt in sorted_elmts])
  # assert(n_cell == nb_elemts)

  elmts_pdm_type, elmts_dn = EZU.collect_pdm_type_and_nelemts(sorted_elmts)
  elmts_connectivities = EZU.collect_connectity(sorted_elmts)

  dmesh_nodal.set_sections(elmts_connectivities, elmts_pdm_type, elmts_dn)

  bc_point_lists = collect_distributed_pl(zone, ['ZoneBC_t/BC_t'])
  delmt_bound_idx, delmt_bound = py_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)
  n_elmt_group = delmt_bound_idx.shape[0] - 1

  #Need an holder to prevent memory delete
  pdm_node = I.createUniqueChild(zone, ':CGNS#DMeshNodal#Bnd', 'UserDefinedData_t')
  I.newDataArray('delmt_bound_idx', delmt_bound_idx, parent=pdm_node)
  I.newDataArray('delmt_bound'    , delmt_bound    , parent=pdm_node)

  dmesh_nodal.set_group_elmt(n_elmt_group, delmt_bound_idx, delmt_bound)

  return dmesh_nodal

