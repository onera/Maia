import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as np

import maia.sids.sids as SIDS
from maia.utils import zone_elements_utils as EZU
from maia.utils.parallel import utils as par_utils

from maia.connectivity import connectivity_transform as CNT
from . import cgns_to_pdm_dmeshnodal as CGNSTOPDM

from maia.distribution.distribution_function import create_distribution_node_from_distrib

import Pypdm.Pypdm as PDM


def pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim):
  """
  """

  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()

  if(extract_dim == 2):
    dface_cell_idx, dface_cell = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_EDGE_FACE)
    dface_vtx_idx, dface_vtx   = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX )
    dcell_face_idx, dcell_face = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_EDGE)
    distrib_face               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_EDGE)
    distrib_cell               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_FACE)
  else :
    dface_cell_idx, dface_cell = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_CELL)
    dface_vtx_idx, dface_vtx   = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX )
    dcell_face_idx, dcell_face = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    distrib_face              = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_FACE)
    distrib_cell              = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_CELL)

  # print("dface_cell_idx::", dface_cell_idx)
  # print("dface_cell::"    , dface_cell)

  # print("dface_vtx_idx::", dface_vtx_idx)
  # print("dface_vtx::"    , dface_vtx)

  # print("dcell_face_idx::", dcell_face_idx)
  # print("dcell_face::"    , dcell_face)

  # print("distrib_face::", distrib_face)
  # print("distrib_cell::", distrib_cell)
  # print("n_face::", n_face)
  # print("n_cell::", n_cell)

  n_face  = distrib_face[n_rank]
  if(distrib_face[0] == 1):
    distrib_face = distrib_face-1
    n_face -=1
  dn_face = distrib_face[i_rank+1] - distrib_face[i_rank]

  n_cell  = distrib_cell[n_rank]
  if(distrib_cell[0] == 1):
    distrib_cell = distrib_cell-1
    n_cell -=1
  dn_cell = distrib_cell[i_rank+1] - distrib_cell[i_rank]

  ldistrib_face = np.empty(3, dtype=distrib_face.dtype)
  ldistrib_face[0] = distrib_face[comm.rank]
  ldistrib_face[1] = distrib_face[comm.rank+1]
  ldistrib_face[2] = n_face

  ldistrib_cell = np.empty(3, dtype=distrib_cell.dtype)
  ldistrib_cell[0] = distrib_cell[comm.rank]
  ldistrib_cell[1] = distrib_cell[comm.rank+1]
  ldistrib_cell[2] = n_cell

  distrib_face_vtx = par_utils.gather_and_shift(dface_vtx_idx[dn_face], comm, np.int32)
  distrib_cell_face = par_utils.gather_and_shift(dcell_face_idx[dn_cell], comm, np.int32)

  ermax   = EZU.get_next_elements_range(zone)

  ngon_n  = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
  ngon_elmt_range = np.empty(2, dtype='int64', order='F')
  ngon_elmt_range[0] = ermax+1
  ngon_elmt_range[1] = ermax+n_face

  pe            = np.empty((dface_cell.shape[0]//2, 2), dtype=dface_cell.dtype, order='F')
  CNT.pdm_face_cell_to_pe_cgns(dface_cell, pe)

  # > Attention overflow I8
  eso_ngon    = np.empty(dface_vtx_idx.shape[0], dtype=dface_vtx.dtype)
  eso_ngon[:] = distrib_face_vtx[i_rank] + dface_vtx_idx[:]

  I.createUniqueChild(ngon_n, 'ElementRange', 'IndexRange_t', ngon_elmt_range)
  I.newDataArray('ElementStartOffset' , eso_ngon , parent=ngon_n)
  I.newDataArray('ElementConnectivity', dface_vtx, parent=ngon_n)
  I.newDataArray('ParentElements'     , pe       , parent=ngon_n)

  ermax   = EZU.get_next_elements_range(zone)
  nfac_n  = I.createUniqueChild(zone, 'NFacElements', 'Elements_t', value=[23,0])
  nfac_elmt_range = np.empty(2, dtype='int64', order='F')
  nfac_elmt_range[0] = ermax+1
  nfac_elmt_range[1] = ermax+n_cell

  eso_nfac    = np.empty(dcell_face_idx.shape[0], dtype=dface_vtx.dtype)
  eso_nfac[:] = distrib_cell_face[i_rank] + dcell_face_idx[:]

  I.createUniqueChild(nfac_n, 'ElementRange', 'IndexRange_t', nfac_elmt_range)
  I.newDataArray('ElementStartOffset' , eso_nfac, parent=nfac_n)
  if(dcell_face is not None):
    I.newDataArray('ElementConnectivity', np.abs(dcell_face), parent=nfac_n)
  else:
    I.newDataArray('ElementConnectivity', np.empty( (0), dtype=eso_ngon.dtype), parent=nfac_n)

  create_distribution_node_from_distrib("Element", ngon_n, ldistrib_face)
  np_distrib_face_vtx = np.array([distrib_face_vtx[i_rank], distrib_face_vtx[i_rank+1], distrib_face_vtx[n_rank]], dtype=pe.dtype)
  create_distribution_node_from_distrib("ElementConnectivity", ngon_n   , np_distrib_face_vtx)

  create_distribution_node_from_distrib("Element", nfac_n, ldistrib_cell)
  np_distrib_cell_face = np.array([distrib_cell_face[i_rank], distrib_cell_face[i_rank+1], distrib_cell_face[n_rank]], dtype=pe.dtype)
  create_distribution_node_from_distrib("ElementConnectivity", nfac_n   , np_distrib_cell_face)

  return ngon_elmt_range[0]-1

def pdm_dmesh_to_cgns(result_dmesh, zone, comm, extract_dim):
  """
  """
  next_ngon = pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim)

  if(extract_dim == 2):
    group_idx, pdm_group = result_dmesh.dmesh_bound_get(PDM._PDM_BOUND_TYPE_EDGE)
  else:
    group_idx, pdm_group = result_dmesh.dmesh_bound_get(PDM._PDM_BOUND_TYPE_FACE)

  #print(group_idx)
  group = np.copy(pdm_group)
  group += next_ngon

  i_group = 0
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for i_bc, bc in enumerate(I.getNodesFromType1(zone_bc, 'BC_t')):
      pr_n = I.getNodeFromName1(bc, 'PointRange')
      pl_n = I.getNodeFromName1(bc, 'PointList')
      if(pr_n):
        I._rmNode(bc, pr_n)
        I._rmNodesByName(bc, 'PointRange#Size')
      if(pl_n):
        I._rmNode(bc, pl_n)
      I.newGridLocation('FaceCenter', parent=bc)
      start, end = group_idx[i_bc], group_idx[i_bc+1]
      dn_face_bnd = end - start
      I.newPointList(value=group[start:end].reshape( (1,dn_face_bnd), order='F' ), parent=bc)

# -----------------------------------------------------------------
def generate_ngon_from_std_elements(dist_tree, comm):
  """
  """
  bases = I.getNodesFromType(dist_tree, 'CGNSBase_t')

  for base in bases:
    base_dim = I.getValue(base)
    extract_dim = base_dim[0]
    #print("extract_dim == ", extract_dim)
    zones_u = [zone for zone in I.getZones(base) if I.getZoneType(zone) == 2]

    n_mesh = len(zones_u)

    dmntodm = PDM.DMeshNodalToDMesh(n_mesh, comm)
    dmesh_nodal_list = list()
    for i_zone, zone in enumerate(zones_u):
      dmn = CGNSTOPDM.cgns_to_pdm(zone, comm)
      dmn.generate_distribution()
      dmntodm.add_dmesh_nodal(i_zone, dmn)

    # PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_FACE
    if(extract_dim == 2):
      dmntodm.compute(PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_EDGE,
                      PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSLATE_GROUP_TO_EDGE)
    else:
      dmntodm.compute(PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_FACE,
                      PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSLATE_GROUP_TO_FACE)

    dmntodm.transform_to_coherent_dmesh(extract_dim)

    for i_zone, zone in enumerate(zones_u):
      result_dmesh = dmntodm.get_dmesh(i_zone)
      pdm_dmesh_to_cgns(result_dmesh, zone, comm, extract_dim)

      # > Remove internal holder state
      I._rmNodesByName(zone, ':CGNS#DMeshNodal#Bnd')

  # > Generate correctly zone_grid_connectivity
