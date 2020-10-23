import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY

import maia.sids.sids as SIDS
from maia.utils import zone_elements_utils as EZU

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
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  dvtx_coord  = NPY.empty(n_vtx*3, dtype='double', order='C')
  print("TODO REMOVE COORDINATE in generate_ngon_from_std_elements.pyx")
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > First sort it
  sort_elmt_3d, sort_elmt_2d, sort_elmt_1d = EZU.get_ordered_elements_std(zone)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Manage case
  if(len(sort_elmt_3d) == 0):
    sort_elmt_vol  = sort_elmt_2d
    sort_elmt_surf = sort_elmt_1d
  else:
    sort_elmt_vol  = sort_elmt_3d
    sort_elmt_surf = sort_elmt_2d
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Sanity check and coolect type and n_elmts
  nb_elemts = 0;
  for elmt in sort_elmt_vol:
    elmt_range  = I.getNodeFromName(elmt, "ElementRange")
    nb_elemts  += elmt_range[1][1]-elmt_range[1][0]+1
  assert(n_cell == nb_elemts)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  sort_elmt_vol_type, sort_elmt_vol_n_elmt = EZU.collect_pdm_type_and_nelemts(sort_elmt_vol)
  sort_elmt_connect = EZU.collect_connectity(sort_elmt_vol)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  print("nb_elemts::", nb_elemts)
  print("n_cell   ::", n_cell)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  distributed_mesh_nodal = PDM.DistributedMeshNodal(comm, n_vtx, nb_elemts)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  distributed_mesh_nodal.SetCoordinnates(dvtx_coord)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # print("type(sort_elmt_vol)       ::", type(sort_elmt_vol))
  # print("type(sort_elmt_vol_type)  ::", type(sort_elmt_vol_type))
  # print("type(sort_elmt_vol_n_elmt)::", type(sort_elmt_vol_n_elmt))
  # print("sort_elmt_connect   ::", sort_elmt_connect)
  # print("sort_elmt_vol_type  ::", sort_elmt_vol_type)
  # print("sort_elmt_vol_n_elmt::", sort_elmt_vol_n_elmt)
  distributed_mesh_nodal.SetSections(sort_elmt_connect,
                                     sort_elmt_vol_type,
                                     sort_elmt_vol_n_elmt)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Compute
  distributed_mesh_nodal.Compute()
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Get
  face_cell_dict    = distributed_mesh_nodal.getFaceCell()
  cell_face_dict    = distributed_mesh_nodal.getCellFace()
  face_vtx_idx_dict = distributed_mesh_nodal.getFaceVtx()
  distrib_face      = distributed_mesh_nodal.getDistribFace()
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  print("face_cell_dict::", face_cell_dict)
  print("cell_face_dict::", cell_face_dict)
  print("face_vtx_idx  ::", face_vtx_idx_dict  )
  print("distrib_face  ::", distrib_face  )
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # > Volumic is ok, we need to identifie and translate now boundary conditions
  #   with the new face numbering
  # PDM.ParentElementFind()
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # >
  n_face  = face_cell_dict['sFace']
  dn_face = face_cell_dict['dNFace']

  ermax   = EZU.get_next_elements_range(zone)

  ngon_n  = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
  ngon_elmt_range = NPY.empty(2, dtype='int64', order='F')
  ngon_elmt_range[0] = ermax
  ngon_elmt_range[1] = ermax+n_face

  I.createUniqueChild(ngon_n, 'ElementRange', 'IndexRange_t', ngon_elmt_range)
  I.newDataArray('ElementStartOffset' , face_vtx_idx_dict["npdFaceVtxIdx"], parent=ngon_n)
  I.newDataArray('ElementConnectivity', face_vtx_idx_dict["npdFaceVtx"]   , parent=ngon_n)
  I.newDataArray('ParentElements'     , face_cell_dict["npdFaceCell"]     , parent=ngon_n)
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------
def generate_ngon_from_std_elements(dist_tree, comm):
  """
  """
  zones_u = [zone for zone in I.getZones(dist_tree) if I.getZoneType(zone) == 2]

  for zone in zones_u:
    generate_ngon_from_std_elements_zone(zone, comm)

  # > Generate correctly zone_grid_connectivity
