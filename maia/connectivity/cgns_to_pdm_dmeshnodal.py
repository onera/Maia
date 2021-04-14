import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY

import maia.sids.sids as SIDS
import maia.sids.Internal_ext as IE
from maia.utils import zone_elements_utils as EZU

import Pypdm.Pypdm as PDM


def concatenate_bc(zone):
  """
  """
  n_elmt_group = 0
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    bcs = I.getNodesFromType1(zone_bc, 'BC_t')
    n_elmt_group += len(bcs)

  delmt_bound        = list()
  delmt_bound_idx    = NPY.empty(n_elmt_group+1, dtype='int32', order='F')
  delmt_bound_idx[0] = 0
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    bcs = I.getNodesFromType1(zone_bc, 'BC_t')
    for i_group, bc in enumerate(bcs):
      pl_n = I.getNodeFromName1(bc, 'PointList')
      if pl_n is not None:
        # > Don't use I.getValue which return an int instead of np array if len(PL)=1
        pl = pl_n[1][0,:]
      else:
        pr_n = I.getNodeFromName1(bc, 'PointRange')
        if(pr_n is not None):
          pr = I.getValue(pr_n)
          distrib        = IE.getDistribution(bc, 'Index')
          pl = NPY.arange(pr[0][0]+distrib[0], pr[0][0]+distrib[1], dtype=pr.dtype)
        else:
          pl = NPY.empty(0, dtype='int32', order='F')
      delmt_bound.append(pl)
      delmt_bound_idx[i_group+1] = delmt_bound_idx[i_group] + pl.shape[0]

  if n_elmt_group > 0:
    delmt_bound = NPY.concatenate(delmt_bound)
  else:
    delmt_bound = None

  #print(n_elmt_group, delmt_bound_idx, delmt_bound)

  # > Holder state
  pdm_node = I.createUniqueChild(zone, ':CGNS#DMeshNodal#Bnd', 'UserDefinedData_t')
  I.newDataArray('delmt_bound_idx', delmt_bound_idx, parent=pdm_node)
  I.newDataArray('delmt_bound'    , delmt_bound    , parent=pdm_node)

  return n_elmt_group, delmt_bound_idx, delmt_bound


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
    # If NGon / NFac skip
    elmt_range  = I.getNodeFromName1(elmt, "ElementRange")
    nb_elemts  += elmt_range[1][1]-elmt_range[1][0]+1
  # assert(n_cell == nb_elemts)

  elmt_by_ascending_gnum_type, elmt_by_ascending_gnum_n_elmt = EZU.collect_pdm_type_and_nelemts(elmt_by_ascending_gnum)
  sort_elmt_connect = EZU.collect_connectity(elmt_by_ascending_gnum)

  n_elmt_group, delmt_bound_idx, delmt_bound = concatenate_bc(zone)

  distributed_mesh_nodal = PDM.DistributedMeshNodal(comm, n_vtx, n_cell)

  # distributed_mesh_nodal.set_coordinnates(dvtx_coord)
  distributed_mesh_nodal.set_group_elmt(n_elmt_group, delmt_bound_idx, delmt_bound)

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

