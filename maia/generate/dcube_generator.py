import Converter.Internal     as I
import maia.sids.Internal_ext as IE
import maia.sids.sids as SIDS

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.distribution.distribution_function import create_distribution_node_from_distrib
from maia.utils.parallel                     import utils          as par_utils
from maia.sids                               import elements_utils as EU

import maia.distribution.distribution_function as MID
import maia.sids.Internal_ext as IE

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

# --------------------------------------------------------------------------
def dcube_nodal_generate(n_vtx, edge_length, origin, cgns_elmt_type, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube with various type of elements, and
  return a CGNS PyTree
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  t_elmt = EU.cgns_elt_name_to_pdm_element_type(cgns_elmt_type)

  dcube = PDM.DCubeNodalGenerator(n_vtx, edge_length,
                                  origin[0], origin[1], origin[2],
                                  t_elmt,
                                  comm)

  dmesh_nodal = dcube.get_dmesh_nodal()
  g_dims      = dmesh_nodal.dmesh_nodal_get_g_dims()
  sections    = dmesh_nodal.dmesh_nodal_get_sections(comm)
  groups      = dmesh_nodal.dmesh_nodal_get_group()

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[g_dims["n_vtx_abs"], g_dims["n_cell_abs"], 0]],
                        'Unstructured', parent=dist_base)

  # > Grid coordinates
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', sections['vtx']['np_vtx'][0::3], parent=grid_coord)
  I.newDataArray('CoordinateY', sections['vtx']['np_vtx'][1::3], parent=grid_coord)
  I.newDataArray('CoordinateZ', sections['vtx']['np_vtx'][2::3], parent=grid_coord)

  # > Section implicitement range donc on maintiens un compteur
  shift_elmt = 1
  for i_section, section in enumerate(sections["sections"]):
    # print("section = ", section)
    cgns_elmt_type = EU.pdm_elt_name_to_cgns_element_type(section["pdm_type"])
    # print("cgns_elmt_type :", cgns_elmt_type)

    elmt = I.newElements('{}.{}'.format(cgns_elmt_type, i_section), cgns_elmt_type,
                         erange = [shift_elmt, shift_elmt + section["np_distrib"][n_rank]-1], parent=dist_zone)
    I.newDataArray('ElementConnectivity', section["np_connec"], parent=elmt)

    shift_elmt += section["np_distrib"][n_rank]

    distrib   = np.array([section["np_distrib"][i_rank], section["np_distrib"][i_rank+1], section["np_distrib"][n_rank]])
    IE.newDistribution({'Element' : distrib}, parent=elmt)

  # > BCs
  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']
  face_group_n   = np.diff(face_group_idx)

  face_group = groups['dgroup_elmt']
  distri = np.empty(n_rank, dtype=face_group.dtype)
  n_face_group = face_group_idx.shape[0] - 1

  for i_bc in range(n_face_group):
    bc_n = I.newBC('dcube_bnd_{0}'.format(i_bc), btype='BCWall', parent=zone_bc)
    I.newGridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    comm.Allgather(dn_face_bnd, distri)
    r_offset  = sum(distri[:i_rank])
    distrib_n = I.createNode(':CGNS#Distribution', 'UserDefinedData_t', parent=bc_n)
    distrib   = np.array([r_offset, r_offset+dn_face_bnd, sum(distri)], dtype=face_group.dtype)
    I.newDataArray('Index', distrib, parent=distrib_n)

  # > Distributions
  np_distrib_cell = MID.uniform_distribution(g_dims["n_cell_abs"], comm)

  distri_vtx = sections['vtx']['np_vtx_distrib']
  np_distrib_vtx      = np.array([distri_vtx      [i_rank], distri_vtx      [i_rank+1], distri_vtx      [n_rank]], dtype=distri_vtx.dtype)

  create_distribution_node_from_distrib("Cell"               , dist_zone, np_distrib_cell    )
  create_distribution_node_from_distrib("Vertex"             , dist_zone, np_distrib_vtx     )

  return dist_tree
