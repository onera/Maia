import Converter.Internal as I
import numpy              as NPY

def bnd_pdm_to_cgns(zone, dist_zone, comm):
  """
  """
  ppart_ud            = I.getNodeFromName1(zone, ':CGNS#Ppart')
  ipart               = I.getNodeFromName1(ppart_ud, 'ipart')[1][0]
  face_bound          = I.getNodeFromName1(ppart_ud, 'np_face_bound'         )[1]
  face_bound_idx      = I.getNodeFromName1(ppart_ud, 'np_face_bound_idx'     )[1]
  face_bound_ln_to_gn = I.getNodeFromName1(ppart_ud, 'np_face_bound_ln_to_gn')[1]
  print("face_bound = ",face_bound)
  print("face_bound_idx= ",face_bound_idx)
  print("face_bound_ln_to_gn = ",face_bound_ln_to_gn)

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
        pl_data    = NPY.empty((1, pl_size), order='F', dtype=NPY.int32)
        pl_data[0] = NPY.copy(face_bound[beg_pl:end_pl])
        I.newGridLocation('FaceCenter', parent=bc_n)
        I.newPointList(value=pl_data, parent=bc_n)
        I.newDataArray(name='LNtoGN', value=NPY.copy(face_bound_ln_to_gn[beg_pl:end_pl]), parent=bc_n)
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
