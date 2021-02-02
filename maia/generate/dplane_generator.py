import Converter.Internal as I
import maia.sids.sids as SIDS

from maia.distribution.distribution_function import create_distribution_node_from_distrib

import numpy as NPY

import Pypdm.Pypdm as PDM

# --------------------------------------------------------------------------
def dplane_generate(xmin, xmax, ymin, ymax,
                    have_random, init_random,
                    nx, ny,
                    comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dplane_dict = PDM.PolyMeshSurf(xmin, xmax, ymin, ymax, have_random, init_random, nx, ny, comm)

  # for key, val in dplane_dict.items():
  #   print(key, val)

  # > En 2D -> dn_face == dn_cell
  distrib      = NPY.empty(n_rank+1, dtype=NPY.int32)
  distrib[0]   = 0
  distrib[1:]  = comm.allgather(dplane_dict['dn_face'])
  distrib_cell = NPY.cumsum(distrib)

  # > En 2D -> dn_vtx == dn_vtx
  distrib[0]  = 0
  distrib[1:] = comm.allgather(dplane_dict['dn_vtx'])
  distri_vtx  = NPY.cumsum(distrib)

  # > En 2D -> dn_edge == dn_face
  distrib[0]   = 0
  distrib[1:]  = comm.allgather(dplane_dict['dn_edge'])
  distrib_face = NPY.cumsum(distrib)

  # > Connectivity by pair
  distrib[0]       = 0
  distrib[1:]      = comm.allgather( 2 * dplane_dict['dn_edge'] )
  distrib_edge_vtx = NPY.cumsum(distrib)

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[distri_vtx[n_rank], distrib_cell[n_rank], 0]],
                        'Unstructured', parent=dist_base)

  # > Grid coordinates
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', dplane_dict['dvtx_coord'][0::3], parent=grid_coord)
  I.newDataArray('CoordinateY', dplane_dict['dvtx_coord'][1::3], parent=grid_coord)
  I.newDataArray('CoordinateZ', dplane_dict['dvtx_coord'][2::3], parent=grid_coord)

  dplane_dict['dedge_vtx_idx'] = NPY.arange(0, 2*dplane_dict['dn_edge']+1, 2, dtype=dplane_dict['dedge_vtx'].dtype)
  assert dplane_dict['dedge_vtx_idx'].shape[0] == dplane_dict['dn_edge']+1

  # > NGon node
  dn_edge = dplane_dict['dn_edge']

  # > For Offset we have to shift to be global
  if i_rank == n_rank - 1:
    eso = distrib_edge_vtx[i_rank] + dplane_dict['dedge_vtx_idx']
  else:
    eso = distrib_edge_vtx[i_rank] + dplane_dict['dedge_vtx_idx'] #[:dn_edge]
  # print(distrib_edge_vtx)
  # print(eso)

  pe     = dplane_dict['dedge_face'].reshape(dn_edge, 2)
  ngon_n = I.newElements('NGonElements', 'NGON',
                         erange = [1, distrib_face[n_rank]], parent=dist_zone)

  I.newDataArray('ElementConnectivity', dplane_dict['dedge_vtx'], parent=ngon_n)
  I.newDataArray('ElementStartOffset' , eso                     , parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                      , parent=ngon_n)

  # > BCs
  zone_bc = I.newZoneBC(parent=dist_zone)

  edge_group_idx = dplane_dict['dedge_group_idx']
  edge_group_n   = NPY.diff(edge_group_idx)

  edge_group = dplane_dict['dedge_group']
  distri = NPY.empty(n_rank, dtype=edge_group.dtype)

  for i_bc in range(dplane_dict['n_edge_group']):
    bc_n = I.newBC('dplane_bnd_{0}'.format(i_bc), btype='BCWall', parent=zone_bc)
    I.newGridLocation('FaceCenter', parent=bc_n)
    start, end = edge_group_idx[i_bc], edge_group_idx[i_bc+1]
    dn_edge_bnd = end - start
    I.newPointList(value=edge_group[start:end].reshape(1,dn_edge_bnd), parent=bc_n)
    comm.Allgather(dn_edge_bnd, distri)
    r_offset  = sum(distri[:i_rank])
    distrib_n = I.createNode(':CGNS#Distribution', 'UserDefinedData_t', parent=bc_n)
    distrib   = NPY.array([r_offset, r_offset+dn_edge_bnd, sum(distri)], dtype=pe.dtype)
    I.newDataArray('Index', distrib, parent=distrib_n)

  # > Distributions
  np_distrib_cell     = NPY.array([distrib_cell    [i_rank], distrib_cell    [i_rank+1], distrib_cell    [n_rank]], dtype=pe.dtype)
  np_distrib_vtx      = NPY.array([distri_vtx      [i_rank], distri_vtx      [i_rank+1], distri_vtx      [n_rank]], dtype=pe.dtype)
  np_distrib_face     = NPY.array([distrib_face    [i_rank], distrib_face    [i_rank+1], distrib_face    [n_rank]], dtype=pe.dtype)
  np_distrib_edge_vtx = NPY.array([distrib_edge_vtx[i_rank], distrib_edge_vtx[i_rank+1], distrib_edge_vtx[n_rank]], dtype=pe.dtype)

  # print(np_distrib_edge_vtx)
  # exit(2)
  create_distribution_node_from_distrib("Cell"               , dist_zone, np_distrib_cell    )
  create_distribution_node_from_distrib("Vertex"             , dist_zone, np_distrib_vtx     )
  create_distribution_node_from_distrib("Element"            , ngon_n   , np_distrib_face    )
  create_distribution_node_from_distrib("ElementConnectivity", ngon_n   , np_distrib_edge_vtx)

  return dist_tree
