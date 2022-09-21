import numpy as np
import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils

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

  # > En 2D -> dn_face == dn_cell
  distrib_cell     = par_utils.gather_and_shift(dplane_dict['dn_face'],comm, np.int32)
  # > En 2D -> dn_vtx == dn_vtx
  distri_vtx       = par_utils.gather_and_shift(dplane_dict['dn_vtx'],comm, np.int32)
  # > En 2D -> dn_edge == dn_face
  distrib_face     = par_utils.gather_and_shift(dplane_dict['dn_edge'],comm, np.int32)
  # > Connectivity by pair
  distrib_edge_vtx = par_utils.gather_and_shift(2*dplane_dict['dn_edge'],comm, np.int32)

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=2, phy_dim=2, parent=dist_tree)
  dist_zone = PT.new_Zone('zone', size=[[distri_vtx[n_rank], distrib_cell[n_rank], 0]],
                        type='Unstructured', parent=dist_base)

  # > Grid coordinates
  coords = {'CoordinateX' : dplane_dict['dvtx_coord'][0::3], 'CoordinateY' : dplane_dict['dvtx_coord'][1::3]}
  grid_coord = PT.new_GridCoordinates(fields=coords, parent=dist_zone)
  assert np.max(np.abs(dplane_dict['dvtx_coord'][2::3])) < 1E-16 #In 2D, this one should be zero

  dplane_dict['dedge_vtx_idx'] = np.arange(0, 2*dplane_dict['dn_edge']+1, 2, dtype=dplane_dict['dedge_vtx'].dtype)
  assert dplane_dict['dedge_vtx_idx'].shape[0] == dplane_dict['dn_edge']+1

  # > NGon node
  dn_edge = dplane_dict['dn_edge']

  # > For Offset we have to shift to be global
  eso = distrib_edge_vtx[i_rank] + dplane_dict['dedge_vtx_idx']

  pe     = dplane_dict['dedge_face'].reshape(dn_edge, 2)
  np_utils.shift_nonzeros(pe, distrib_face[n_rank])
  ngon_n = PT.new_NGonElements('NGonElements', erange = [1, distrib_face[n_rank]], 
      eso = eso, ec = dplane_dict['dedge_vtx'], pe = pe, parent=dist_zone)

  # > BCs
  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  edge_group_idx = dplane_dict['dedge_group_idx']
  edge_group_n   = np.diff(edge_group_idx)

  edge_group = dplane_dict['dedge_group']

  for i_bc in range(dplane_dict['n_edge_group']):
    bc_n = PT.new_BC('dplane_bnd_{0}'.format(i_bc), type='BCWall', parent=zone_bc)
    PT.new_GridLocation('FaceCenter', parent=bc_n)
    start, end = edge_group_idx[i_bc], edge_group_idx[i_bc+1]
    dn_edge_bnd = end - start
    PT.new_PointList(value=edge_group[start:end].reshape(1,dn_edge_bnd), parent=bc_n)

    bc_distrib = par_utils.gather_and_shift(dn_edge_bnd, comm, edge_group.dtype)
    distrib   = np.array([bc_distrib[i_rank], bc_distrib[i_rank+1], bc_distrib[n_rank]])
    MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  np_distrib_cell     = np.array([distrib_cell    [i_rank], distrib_cell    [i_rank+1], distrib_cell    [n_rank]], dtype=pe.dtype)
  np_distrib_vtx      = np.array([distri_vtx      [i_rank], distri_vtx      [i_rank+1], distri_vtx      [n_rank]], dtype=pe.dtype)
  np_distrib_face     = np.array([distrib_face    [i_rank], distrib_face    [i_rank+1], distrib_face    [n_rank]], dtype=pe.dtype)
  np_distrib_edge_vtx = np.array([distrib_edge_vtx[i_rank], distrib_edge_vtx[i_rank+1], distrib_edge_vtx[n_rank]], dtype=pe.dtype)

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=dist_zone)
  MT.newDistribution({'Element' : np_distrib_face, 'ElementConnectivity' : np_distrib_edge_vtx}, parent=ngon_n)

  return dist_tree
