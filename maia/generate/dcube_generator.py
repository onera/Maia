import Converter.Internal     as I
import maia.sids.Internal_ext as IE
import maia.sids.sids as SIDS

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.distribution.distribution_function import create_distribution_node_from_distrib
from maia.utils.parallel import utils as par_utils

import numpy as np

import Pypdm.Pypdm as PDM

# --------------------------------------------------------------------------
def dcube_generate(n_vtx, edge_length, origin, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dcube = PDM.DCubeGenerator(n_vtx, edge_length, *origin, comm)

  dcube_dims = dcube.dcube_dim_get()
  dcube_val  = dcube.dcube_val_get()

  distrib_cell    = par_utils.gather_and_shift(dcube_dims['dn_cell'],   comm, pdm_gnum_dtype)
  distrib_vtx     = par_utils.gather_and_shift(dcube_dims['dn_vtx'],    comm, pdm_gnum_dtype)
  distrib_face    = par_utils.gather_and_shift(dcube_dims['dn_face'],   comm, pdm_gnum_dtype)
  distrib_facevtx = par_utils.gather_and_shift(dcube_dims['sface_vtx'], comm, pdm_gnum_dtype)

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[distrib_vtx[n_rank], distrib_cell[n_rank], 0]],
                        'Unstructured', parent=dist_base)

  # > Grid coordinates
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', dcube_val['dvtx_coord'][0::3], parent=grid_coord)
  I.newDataArray('CoordinateY', dcube_val['dvtx_coord'][1::3], parent=grid_coord)
  I.newDataArray('CoordinateZ', dcube_val['dvtx_coord'][2::3], parent=grid_coord)

  # > NGon node
  dn_face = dcube_dims['dn_face']

  # > For Offset we have to shift to be global
  eso = distrib_facevtx[i_rank] + dcube_val['dface_vtx_idx']

  pe     = dcube_val['dface_cell'].reshape(dn_face, 2)
  ngon_n = I.newElements('NGonElements', 'NGON',
                         erange = [1, distrib_face[n_rank]], parent=dist_zone)

  I.newDataArray('ElementConnectivity', dcube_val['dface_vtx'], parent=ngon_n)
  I.newDataArray('ElementStartOffset' , eso                   , parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                    , parent=ngon_n)

  # > BCs
  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = dcube_val['dface_group_idx']
  face_group_n   = np.diff(face_group_idx)

  face_group = dcube_val['dface_group']

  for i_bc in range(dcube_dims['n_face_group']):
    bc_n = I.newBC('dcube_bnd_{0}'.format(i_bc), btype='BCWall', parent=zone_bc)
    I.newGridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)

    bc_distrib = par_utils.gather_and_shift(dn_face_bnd, comm, pdm_gnum_dtype)
    distrib   = np.array([bc_distrib[i_rank], bc_distrib[i_rank+1], bc_distrib[n_rank]])
    IE.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  np_distrib_cell    = np.array([distrib_cell[i_rank]   , distrib_cell[i_rank+1]   , distrib_cell[n_rank]]   )
  np_distrib_vtx     = np.array([distrib_vtx[i_rank]    , distrib_vtx[i_rank+1]    , distrib_vtx[n_rank]]    )
  np_distrib_face    = np.array([distrib_face[i_rank]   , distrib_face[i_rank+1]   , distrib_face[n_rank]]   )
  np_distrib_facevtx = np.array([distrib_facevtx[i_rank], distrib_facevtx[i_rank+1], distrib_facevtx[n_rank]])

  create_distribution_node_from_distrib("Cell"               , dist_zone, np_distrib_cell   )
  create_distribution_node_from_distrib("Vertex"             , dist_zone, np_distrib_vtx    )
  create_distribution_node_from_distrib("Element"            , ngon_n   , np_distrib_face   )
  create_distribution_node_from_distrib("ElementConnectivity", ngon_n   , np_distrib_facevtx)

  return dist_tree
