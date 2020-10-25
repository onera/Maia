import Converter.Internal as I
import maia.sids.sids as SIDS


# --------------------------------------------------------------------------
def dcube_generate(n_vtx, edge_length, origin, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  # ************************************************************************
  # > Declaration
  # cdef NPY.ndarray[NPY.int32_t, ndim=1, mode='fortran'] distri
  # ************************************************************************

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Generation from Paradigm
  dcube = PDM.DCubeGenerator(n_vtx, edge_length,
                             origin[0], origin[1], origin[2], comm)

  dcube_dims = dcube.dcube_dim_get()
  dcube_val  = dcube.dcube_val_get()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Recompute distributions
  distrib      = NPY.empty(n_rank+1, dtype=NPY.int32)
  distrib[0]   = 0
  distrib[1:]  = comm.allgather(dcube_dims['dn_cell'])
  distrib_cell = NPY.cumsum(distri)

  distrib[0]  = 0
  distrib[1:] = comm.allgather(dcube_dims['dn_vtx'])
  distri_vtx  = NPY.cumsum(distri)

  distrib[0]   = 0
  distrib[1:]  = comm.allgather(dcube_dims['dn_face'])
  distrib_face = NPY.cumsum(distri)

  distrib[0]       = 0
  distrib[1:]      = comm.allgather(dcube_dims['dface_vtx'])
  distrib_face_vtx = NPY.cumsum(distri)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[distri_vtx[n_rank], distrib_cell[n_rank], 0]],
                        'Unstructured', parent=dist_base)

  # > Grid connectivity
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', dcube_val['dvtx_coord'][0::3], parent=grid_coord)
  I.newDataArray('CoordinateY', dcube_val['dvtx_coord'][1::3], parent=grid_coord)
  I.newDataArray('CoordinateZ', dcube_val['dvtx_coord'][2::3], parent=grid_coord)

  # > NGon node
  dn_face = dcube_dims['dn_face']

  #For Offset we have to shift to be global
  if i_rank == n_rank - 1:
    eso = distrib_face_vtx[i_rank] + dcube_val['dface_vtx_idx']
  else:
    eso = distrib_face_vtx[i_rank] + dcube_val['dface_vtx_idx'][:dn_face]

  pe     = dcube_val['dFaceCell'].reshape(dn_face, 2)
  ngon_n = I.newElements('NGonElements', 'NGON',
                         erange = [1, distrib_face[n_rank]], parent=dist_zone)

  I.newDataArray('ElementConnectivity', dcube_val['dface_vtx'], parent=ngon_n)
  I.newDataArray('ElementStartOffset' , eso                   , parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                    , parent=ngon_n)

  # > BCs
  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = dcube_val['dface_group_idx']
  face_group_n   = NPY.diff(face_group_idx)

  face_group = dcube_val['dface_group']
  distri = NPY.empty(n_rank, dtype=NPY.int32)

  for i_bc in range(dcube_dims['nface_group']):
    bc_n = I.newBC('dcube_bnd_{0}'.format(i_bc), parent=zone_bc)
    I.newGridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    comm.Allgather(dn_face_bnd, distri)
    r_offset  = sum(distri[:i_rank])
    distrib_n = I.createNode(':CGNS#Distribution', 'UserDefinedData_t', parent=bc_n)
    distrib   = NPY.array([r_offset, r_offset+dn_face_bnd, sum(distri)], dtype=NPY.int32)
    I.newDataArray('Distribution', distrib, parent=distrib_n)

  # > Distributions
  distrib_n = I.createNode(':CGNS#Distribution', 'UserDefinedData_t', parent=dist_zone)
  distrib  = NPY.array([distrib_cell[i_rank], distrib_cell[i_rank+1], distrib_cell[n_rank]], dtype=NPY.int32)
  I.newDataArray('Distribution_cell', distrib, parent=distrib_n)
  distrib  = NPY.array([distrib_face[i_rank], distrib_face[i_rank+1], distrib_face[n_rank]], dtype=NPY.int32)
  I.newDataArray('Distribution_face', distrib, parent=distrib_n)
  distrib  = NPY.array([distri_vtx[i_rank], distri_vtx[i_rank+1], distri_vtx[n_rank]], dtype=NPY.int32)
  I.newDataArray('Distribution_vtx', distrib, parent=distrib_n)
  distrib  = NPY.array([distrib_face_vtx[i_rank], distrib_face_vtx[i_rank+1], distrib_face_vtx[n_rank]], dtype=NPY.int32)
  I.newDataArray('Distribution_faceVtx', distrib, parent=distrib_n)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  return dist_tree
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
