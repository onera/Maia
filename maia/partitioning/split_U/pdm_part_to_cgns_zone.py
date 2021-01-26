import Converter.Internal as I
import numpy              as np

from maia.connectivity import connectivity_transform as CNT

def zgc_original_pdm_to_cgns(zone, dist_zone, comm):
  """
  Already exist in initial configuration
  """

def zgc_created_pdm_to_cgns(zone, dist_zone, comm, entity='face', zgc_name='ZoneGridConnectivity'):
  """
  Create by splitting
  """
  if(entity == 'face'):
    grid_loc = 'FaceCenter'
  elif(entity == 'vtx'):
    grid_loc = 'Vertex'
  else:
    raise NotImplementedError("Unvalid specified entity")

  dist_zone_name = dist_zone[0]

  ppart_ud                   = I.getNodeFromName1(zone, ':CGNS#Ppart')
  ipart                      = I.getNodeFromName1(ppart_ud, 'ipart')[1][0]
  entity_part_bound_proc_idx = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound_proc_idx'.format(entity))[1]
  entity_part_bound_part_idx = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound_part_idx'.format(entity))[1]
  entity_part_bound_tmp      = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound'         .format(entity))[1]

  entity_part_bound = entity_part_bound_tmp.reshape((4, entity_part_bound_tmp.shape[0]//4), order='F')
  entity_part_bound = entity_part_bound.transpose()

  zgc_n = I.newZoneGridConnectivity(name=zgc_name, parent=zone)

  n_internal_join = entity_part_bound_part_idx.shape[0]-1
  for i_join in range(n_internal_join):

    beg_pl = entity_part_bound_part_idx[i_join  ]
    end_pl = entity_part_bound_part_idx[i_join+1]

    if( beg_pl != end_pl):

      pl_size = end_pl - beg_pl
      pl      = np.empty((1, pl_size), order='F', dtype=np.int32)
      pl[0]   = np.copy(entity_part_bound[beg_pl:end_pl, 0])

      pld    = np.empty((1, pl_size), order='F', dtype=np.int32)
      pld[0] = np.copy(entity_part_bound[beg_pl:end_pl, 3])

      connect_proc = entity_part_bound[beg_pl, 1]
      connect_part = entity_part_bound[beg_pl, 2]-1

      join_n = I.newGridConnectivity(name      = 'JN.P{0}.N{1}.LT.P{2}.N{3}'.format(comm.Get_rank(), ipart, connect_proc, connect_part, i_join),
                                     donorName = dist_zone_name+'.P{0}.N{1}'.format(connect_proc, connect_part),
                                     ctype     = 'Abutting1to1',
                                     parent    = zgc_n)

      I.newGridLocation(grid_loc, parent=join_n)
      I.newPointList(name='PointList'     , value=pl , parent=join_n)
      I.newPointList(name='PointListDonor', value=pld, parent=join_n)

def bnd_pdm_to_cgns(zone, dist_zone, comm):
  """
  """
  ppart_ud            = I.getNodeFromName1(zone, ':CGNS#Ppart')
  ipart               = I.getNodeFromName1(ppart_ud, 'ipart')[1][0]
  face_bound          = I.getNodeFromName1(ppart_ud, 'np_face_bound'         )[1]
  face_bound_idx      = I.getNodeFromName1(ppart_ud, 'np_face_bound_idx'     )[1]
  face_bound_ln_to_gn = I.getNodeFromName1(ppart_ud, 'np_face_bound_ln_to_gn')[1]

  if(face_bound_idx is None):
    return

  for dist_zone_bc in I.getNodesFromType1(dist_zone, 'ZoneBC_t'):
    part_zone_bc = I.newZoneBC(parent = zone)
    for i_bc, dist_bc in enumerate(I.getNodesFromType1(dist_zone_bc, 'BC_t')):
      beg_pl = face_bound_idx[i_bc  ]
      end_pl = face_bound_idx[i_bc+1]

      if( beg_pl != end_pl):
        bcname = dist_bc[0]+'.P{0}.N{1}'.format(comm.Get_rank(), ipart)
        bctype = I.getValue(dist_bc)
        bc_n   = I.newBC(name=bcname, btype=bctype, parent=part_zone_bc)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pl_size = end_pl - beg_pl
        pl_data    = np.empty((1, pl_size), order='F', dtype=np.int32)
        pl_data[0] = np.copy(face_bound[beg_pl:end_pl])
        I.newGridLocation('FaceCenter', parent=bc_n)
        I.newPointList(value=pl_data, parent=bc_n)
        I.newDataArray(name='LNtoGN', value=np.copy(face_bound_ln_to_gn[beg_pl:end_pl]), parent=bc_n)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Recuperation of UserDefinedData and FamilyName in DistTree
        fam_dist_bc = I.getNodeFromType1(dist_bc, 'FamilyName_t')
        if(fam_dist_bc is not None):
          I._addChild(bc_n, fam_dist_bc)
        solver_prop = I.getNodeFromName1(dist_bc, '.Solver#BC')
        if(solver_prop is not None):
          I._addChild(bc_n, solver_prop)
        boundary_marker = I.getNodeFromName1(dist_bc, 'BoundaryMarker')
        if(boundary_marker is not None):
          I._addChild(bc_n, boundary_marker)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def save_in_tree_part_info(zone, dims, data, comm):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  ppart_node = I.createUniqueChild(zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for k in dims.keys():
    I.newDataArray(k, dims[k], parent=ppart_node)

  for k in data.keys():
    if (type(data[k])==np.ndarray):
      I.newDataArray(k, np.copy(data[k]), parent=ppart_node)

  I.newDataArray('iproc', i_rank, parent=ppart_node)

def pdm_vtx_to_cgns_grid_coordinates(zone, dims, data):
  """
  """
  grid_c = I.newGridCoordinates(parent=zone)
  I.newDataArray('CoordinateX', data['np_vtx_coord'][0::3], parent=grid_c)
  I.newDataArray('CoordinateY', data['np_vtx_coord'][1::3], parent=grid_c)
  I.newDataArray('CoordinateZ', data['np_vtx_coord'][2::3], parent=grid_c)

def pdm_elmt_to_cgns_elmt(zone, dims, data, dist_zone):
  """
  """
  if (dims['n_section']==0):
    pdm_face_cell = data['np_face_cell']
    pe = np.empty((pdm_face_cell.shape[0]//2, 2), dtype=pdm_face_cell.dtype, order='F')
    CNT.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)

    ngon_n = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
    I.newDataArray('ElementConnectivity', data['np_face_vtx']    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , data['np_face_vtx_idx'], parent=ngon_n)
    I.newDataArray('ParentElements'     , pe                     , parent=ngon_n)
    I.createNode('ElementRange', 'IndexRange_t',
                 [1, dims['n_face']], parent=ngon_n)

    nface_n = I.createUniqueChild(zone, 'NFacElements', 'Elements_t', value=[23,0])
    I.newDataArray('ElementConnectivity', data['np_cell_face']    , parent=nface_n)
    I.newDataArray('ElementStartOffset' , data['np_cell_face_idx'], parent=nface_n)
    I.createNode('ElementRange', 'IndexRange_t',
                 [dims['n_face']+1, dims['n_face']+dims['n_cell']], parent=nface_n)
  else:
    elt_section_nodes = I.getNodesFromType(dist_zone,"Elements_t")

    n_section = dims['n_section']
    assert len(elt_section_nodes)==n_section

    n_elt_sum = 0
    for i,elt_section_node in enumerate(elt_section_nodes):
      elt_name = I.getName(elt_section_node)
      elt_type = I.getValue(elt_section_node)[0]
      ngon_n = I.createUniqueChild(zone, elt_name, 'Elements_t', value=[elt_type,0])
      I.newDataArray('ElementConnectivity', data['np_elt_vtx'][i]    , parent=ngon_n)
      I.createNode('ElementRange', 'IndexRange_t', [n_elt_sum+1, n_elt_sum+dims['n_elt'][i]], parent=ngon_n)
      n_elt_sum += dims['n_elt'][i]


def pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm):

  #Dims and data should be related to the dist zone and of size n_parts
  part_zones = list()
  for i_part, (dims, data) in enumerate(zip(l_dims, l_data)):

    part_zone = I.newZone(name  = '{0}.P{1}.N{2}'.format(I.getName(dist_zone), comm.Get_rank(), i_part),
                          zsize = [[dims['n_vtx'],dims['n_cell'],0]],
                          ztype = 'Unstructured')

    save_in_tree_part_info(part_zone, dims, data, comm)
    pdm_vtx_to_cgns_grid_coordinates(part_zone, dims, data)
    pdm_elmt_to_cgns_elmt(part_zone, dims, data, dist_zone)

    bnd_pdm_to_cgns(part_zone, dist_zone, comm)
    # TODO
    #zgc_original_pdm_to_cgns(part_zone, dist_zone, comm)

    # TODO
    #zgc_created_pdm_to_cgns(part_zone, dist_zone, comm, 'face')
    zgc_created_pdm_to_cgns(part_zone, dist_zone, comm, 'vtx', 'ZoneGridConnectivity#Vertex')

    if(data['np_vtx_ghost_information'] is not None):
      vtx_ghost_info = data['np_vtx_ghost_information']
      first_ghost_idx = np.searchsorted(vtx_ghost_info, 2)
      n_ghost_node = len(vtx_ghost_info) - first_ghost_idx
      coord_node = I.getNodeFromName(part_zone,"GridCoordinates")

      I.newUserDefinedData("FSDM#n_ghost",value=[n_ghost_node],parent=coord_node)

    part_zones.append(part_zone)

  return part_zones


def pdm_part_to_cgns_zoneold(zone, dist_zone, dims, data, comm):
  """
  """
  save_in_tree_part_info(zone, dims, data, comm)
  pdm_vtx_to_cgns_grid_coordinates(zone, dims, data)
  pdm_elmt_to_cgns_elmt(zone, dims, data, dist_zone)

  # LYT._convertVtxAndFaceCellForCGNS(zone)
  # if I.getNodeFromName2(dist_zone, 'ElementStartOffset') is not None:
  #   NGonNode = I.getNodeFromName1(zone, 'NGonElements')
  #   EC  = I.getNodeFromName(child, 'np_face_vtx')[1]
  #   ESO = I.getNodeFromName(child, 'np_face_vtx_idx')[1]
  #   I.newDataArray('ElementConnectivity', EC, parent=NGonNode)
  #   I.newDataArray('ElementStartOffset', ESO, parent=NGonNode)
  #   I.createNode('ElementRange', 'IndexRange_t',
  #                [1, dims['n_face']], parent=NGonNode)
  # else:
  #   TBX._convertMorseToNGon2(zone)
  #   TBX._convertMorseToNFace(zone)

  bnd_pdm_to_cgns(zone, dist_zone, comm)
  # TODO
  #zgc_original_pdm_to_cgns(zone, dist_zone, comm)

  # TODO
  #zgc_created_pdm_to_cgns(zone, dist_zone, comm, 'face')
  zgc_created_pdm_to_cgns(zone, dist_zone, comm, 'vtx', 'ZoneGridConnectivity#Vertex')
