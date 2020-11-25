import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY

import maia.sids.sids as SIDS
from maia.utils import zone_elements_utils as EZU

import Pypdm.Pypdm as PDM

def cgns_to_pdm(zone, comm):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  # dvtx_coord  = NPY.empty(n_vtx*3, dtype='double', order='C')
  # print("TODO REMOVE COORDINATE in generate_ngon_from_std_elements.pyx")

  elmt_by_ascending_gnum = EZU.get_ordered_elements_std(zone)

  nb_elemts = 0;
  for elmt in elmt_by_ascending_gnum:
    elmt_range  = I.getNodeFromName1(elmt, "ElementRange")
    nb_elemts  += elmt_range[1][1]-elmt_range[1][0]+1
  # assert(n_cell == nb_elemts)

  elmt_by_ascending_gnum_type, elmt_by_ascending_gnum_n_elmt = EZU.collect_pdm_type_and_nelemts(elmt_by_ascending_gnum)
  sort_elmt_connect = EZU.collect_connectity(elmt_by_ascending_gnum)

  # > Si melangé au poly3d faux !

  # print("nb_elemts::", nb_elemts)
  # print("n_cell   ::", n_cell)

  distributed_mesh_nodal = PDM.DistributedMeshNodal(comm, n_vtx, n_cell)

  # distributed_mesh_nodal.set_coordinnates(dvtx_coord)

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # print("type(elmt_by_ascending_gnum)       ::", type(elmt_by_ascending_gnum))
  # print("type(elmt_by_ascending_gnum_type)  ::", type(elmt_by_ascending_gnum_type))
  # print("type(elmt_by_ascending_gnum_n_elmt)::", type(elmt_by_ascending_gnum_n_elmt))
  # print("sort_elmt_connect   ::", sort_elmt_connect)
  # print("elmt_by_ascending_gnum_type  ::", elmt_by_ascending_gnum_type)
  # print("elmt_by_ascending_gnum_n_elmt::", elmt_by_ascending_gnum_n_elmt)
  distributed_mesh_nodal.set_sections(sort_elmt_connect,
                                      elmt_by_ascending_gnum_type,
                                      elmt_by_ascending_gnum_n_elmt)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  return distributed_mesh_nodal

