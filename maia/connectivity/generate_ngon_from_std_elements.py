import Converter.Internal as I
import numpy              as np

import maia.sids.sids as SIDS
import maia.sids.Internal_ext as IE
from maia.sids import elements_utils as EU
from maia.utils.parallel import utils as par_utils

from maia.connectivity import connectivity_transform as CNT
from maia.connectivity import remove_element as RME
from maia.partitioning.split_U.cgns_to_pdm_dmesh_nodal import cgns_dist_zone_to_pdm_dmesh_nodal

from maia.distribution.distribution_function import create_distribution_node_from_distrib

import Pypdm.Pypdm as PDM


def pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim):
  """
  """

  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()

  # Get PDM output
  cell = "FACE" if extract_dim == 2 else "CELL"
  face = "EDGE" if extract_dim == 2 else "FACE"
  dface_cell_idx, dface_cell = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{face}_{cell}"))
  dface_vtx_idx,  dface_vtx  = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{face}_VTX"))
  dcell_face_idx, dcell_face = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{cell}_{face}"))
  distrib_face               = result_dmesh.dmesh_distrib_get(eval(f"PDM._PDM_MESH_ENTITY_{face}"))
  distrib_cell               = result_dmesh.dmesh_distrib_get(eval(f"PDM._PDM_MESH_ENTITY_{cell}"))
  group_idx, pdm_group       = result_dmesh.dmesh_bound_get(eval(f"PDM._PDM_BOUND_TYPE_{face}"))

  #Shift distribution if starting at 1
  if(distrib_face[0] == 1):
    distrib_face = distrib_face-1
  if(distrib_cell[0] == 1):
    distrib_cell = distrib_cell-1

  dn_face = distrib_face[i_rank+1] - distrib_face[i_rank]
  n_face  = distrib_face[n_rank]

  n_cell  = distrib_cell[n_rank]
  dn_cell = distrib_cell[i_rank+1] - distrib_cell[i_rank]

  ldistrib_face = distrib_face[[i_rank, i_rank+1, n_rank]]
  ldistrib_cell = distrib_cell[[i_rank, i_rank+1, n_rank]]

  distrib_face_vtx  = par_utils.gather_and_shift(dface_vtx_idx[dn_face], comm, np.int32)
  distrib_cell_face = par_utils.gather_and_shift(dcell_face_idx[dn_cell], comm, np.int32)

  # Create NGon node
  ermax   = max([SIDS.ElementRange(e)[1] for e in I.getNodesFromType1(zone, 'Elements_t')])

  pe = np.empty((dface_cell.shape[0]//2, 2), dtype=dface_cell.dtype, order='F')
  CNT.pdm_face_cell_to_pe_cgns(dface_cell, pe)
  #NGon PE must refer to nFace indexes, we have to shift
  pe[np.where(pe != 0)] += ermax+n_face
  # > Attention overflow I8
  eso_ngon = dface_vtx_idx + distrib_face_vtx[i_rank]

  ngon_n  = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
  I.createUniqueChild(ngon_n, 'ElementRange', 'IndexRange_t', [ermax+1, ermax+n_face])
  I.newDataArray('ElementStartOffset' , eso_ngon , parent=ngon_n)
  I.newDataArray('ElementConnectivity', dface_vtx, parent=ngon_n)
  I.newDataArray('ParentElements'     , pe       , parent=ngon_n)

  create_distribution_node_from_distrib("Element", ngon_n, ldistrib_face)
  create_distribution_node_from_distrib("ElementConnectivity", ngon_n, distrib_face_vtx[[i_rank, i_rank+1, n_rank]])

  # Create NFace node
  ermax   = max([SIDS.ElementRange(e)[1] for e in I.getNodesFromType1(zone, 'Elements_t')])
  eso_nfac = dcell_face_idx + distrib_cell_face[i_rank]

  nfac_n  = I.createUniqueChild(zone, 'NFaceElements', 'Elements_t', value=[23,0])
  I.createUniqueChild(nfac_n, 'ElementRange', 'IndexRange_t', [ermax+1, ermax+n_cell])
  I.newDataArray('ElementStartOffset' , eso_nfac, parent=nfac_n)
  if dcell_face is not None:
    I.newDataArray('ElementConnectivity', np.abs(dcell_face), parent=nfac_n)
  else:
    I.newDataArray('ElementConnectivity', np.empty(0, dtype=eso_ngon.dtype), parent=nfac_n)

  create_distribution_node_from_distrib("Element", nfac_n, ldistrib_cell)
  create_distribution_node_from_distrib("ElementConnectivity", nfac_n, distrib_cell_face[[i_rank, i_rank+1, n_rank]])

  #Manage BCs : shift PL values to reach refer ngon_elements
  group = np.copy(pdm_group) + (EU.get_range_of_ngon(zone)[0]-1)
  for i_bc, bc in enumerate(IE.getNodesFromTypeMatching(zone, 'ZoneBC_t/BC_t')):
    I._rmNodesByName(bc, 'PointRange')
    I._rmNodesByName(bc, 'PointList')
    start, end = group_idx[i_bc], group_idx[i_bc+1]
    I.newPointList(value=group[start:end].reshape((1,-1), order='F'), parent=bc)


# -----------------------------------------------------------------
def compute_ngon_from_std_elements(dist_tree, comm):
  """
  """
  for base in I.getNodesFromType(dist_tree, 'CGNSBase_t'):
    extract_dim = I.getValue(base)[0]
    #print("extract_dim == ", extract_dim)
    zones_u = [zone for zone in I.getZones(base) if SIDS.ZoneType(zone) == "Unstructured"]

    dmn_to_dm = PDM.DMeshNodalToDMesh(len(zones_u), comm)
    dmesh_nodal_list = list()
    for i_zone, zone in enumerate(zones_u):
      dmn = cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm, needs_vertex=False)
      dmn.generate_distribution()
      dmn_to_dm.add_dmesh_nodal(i_zone, dmn)

    # PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_FACE
    face = "EDGE" if extract_dim == 2 else "FACE"
    dmn_to_dm.compute(eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_{face}"),
                    eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSLATE_GROUP_TO_{face}"))

    dmn_to_dm.transform_to_coherent_dmesh(extract_dim)

    for i_zone, zone in enumerate(zones_u):
      result_dmesh = dmn_to_dm.get_dmesh(i_zone)
      pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim)

      # > Remove internal holder state
      I._rmNodesByName(zone, ':CGNS#DMeshNodal#Bnd')

  # > Generate correctly zone_grid_connectivity

def generate_ngon_from_std_elements(dist_tree, comm):
  """
  Generate the ngon and nface elements using compute_ngon_from_std_elements,
  and remove the standard elements from the tree
  Possible optimisation : remove all the element at the same time
  instead of looping
  """
  compute_ngon_from_std_elements(dist_tree,comm)
  for zone in I.getZones(dist_tree):
    elts_to_remove = [elt for elt in I.getNodesFromType1(zone, 'Elements_t') if\
        SIDS.ElementCGNSName(elt) not in ["NGON_n", "NFACE_n"]]
    #2D element should be removed first, to avoid probleme coming from ParentElements
    for elt in sorted(elts_to_remove, key = SIDS.ElementDimension):
      RME.remove_element(zone, elt)
