import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as np

import maia.sids.sids as SIDS
from maia.utils import zone_elements_utils as EZU

from . import cgns_to_pdm_dmeshnodal as CGNSTOPDM

from maia.distribution.distribution_function import create_distribution_node_from_distrib

import Pypdm.Pypdm as PDM


# def replace_std_elements_by_ngon(dist_tree, comm):
#   """
#   """
#   generate_ngon_from_std_elements(dist_tree, comm)

#   # > Suppress std_elements and reput er ngon = [1, nface] et nface = [nface+1, ncell], elt = void

# def add_ngon(dist_tree, comm):
#   """
#   """
#   generate_ngon_from_std_elements(dist_tree, comm)
#   # > Organize std + ngon +

# def add_ngon_and_nface(dist_tree, comm):
#   """
#   """
#   generate_ngon_from_std_elements(dist_tree, comm)


# def super_fonction(zone, comm):
#   """
#   """
#   zones_u = [zone for zone in I.getZones(dist_tree) if I.getZoneType(zone) == 2]
#   for zone in zones_u:
#     pdm_dmn = setup_pdm_distributed_mesh_nodal(zone, comm)
#     # > Recollecment des frontières
#     pdm_dmn.service1()
#     pdm_dmn.service2()

#   # > Rebuild ZoneGridConnecitivty -- Update PointListDonor

#   return pdm_dmn

# def setup_pdm_distributed_mesh_nodal(zone, comm):
#   """
#   """
#   return pdm_dmn


def generate_ngon_from_std_elements_zone(zone, comm):
  """
  """
  distributed_mesh_nodal = CGNSTOPDM.cgns_to_pdm(zone, comm)
  distributed_mesh_nodal.compute()

  face_cell_dict    = distributed_mesh_nodal.get_face_cell()
  cell_face_dict    = distributed_mesh_nodal.get_cell_face()
  face_vtx_idx_dict = distributed_mesh_nodal.get_face_vtx()
  distrib_face      = distributed_mesh_nodal.get_distrib_face()

  # print("face_cell_dict::", face_cell_dict)
  # print("cell_face_dict::", cell_face_dict)
  # print("face_vtx_idx  ::", face_vtx_idx_dict  )
  # print("distrib_face  ::", distrib_face  )

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Volumic is ok, we need to identifie and translate now boundary conditions
  #   with the new face numbering
  # PDM.ParentElementFind()
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # >
  n_face  = face_cell_dict['n_face']
  dn_face = face_cell_dict['dn_face']

  ermax   = EZU.get_next_elements_range(zone)

  ngon_n  = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
  ngon_elmt_range = np.empty(2, dtype='int64', order='F')
  ngon_elmt_range[0] = ermax
  ngon_elmt_range[1] = ermax+n_face

  pe = face_cell_dict["np_dface_cell"].reshape((n_face,2)) # TODO AFAIK, this is incoherent with the SIDS, but seems to be used consistently here...
  #pe = face_cell_dict["np_dface_cell"].reshape((2,n_face))
  #pe = np.transpose(pe)
  I.createUniqueChild(ngon_n, 'ElementRange', 'IndexRange_t', ngon_elmt_range)
  I.newDataArray('ElementStartOffset' , face_vtx_idx_dict["np_dface_vtx_idx"], parent=ngon_n)
  I.newDataArray('ElementConnectivity', face_vtx_idx_dict["np_dface_vtx"]   , parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                                  , parent=ngon_n)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ldistrib_face = np.empty(3, dtype=distrib_face.dtype)
  ldistrib_face[0] = distrib_face[comm.rank]
  ldistrib_face[1] = distrib_face[comm.rank+1]
  ldistrib_face[2] = n_face
  create_distribution_node_from_distrib("Distribution", ngon_n, ldistrib_face)

  distrib_face_vtx = np.empty(2, dtype=distrib_face.dtype)
  distrib_face_vtx[0] = face_vtx_idx_dict["np_dface_vtx_idx"][0]
  distrib_face_vtx[1] = face_vtx_idx_dict["np_dface_vtx_idx"][-1]
  create_distribution_node_from_distrib("DistributionElementConnectivity", ngon_n, distrib_face_vtx)

# -----------------------------------------------------------------
def generate_ngon_from_std_elements(dist_tree, comm):
  """
  """
  zones_u = [zone for zone in I.getZones(dist_tree) if I.getZoneType(zone) == 2]

  for zone in zones_u:
    generate_ngon_from_std_elements_zone(zone, comm)

  # > Generate correctly zone_grid_connectivity
