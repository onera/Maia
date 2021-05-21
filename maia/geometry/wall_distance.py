
__doc__ = """
CGNS python module which interface the ParaDiGM library for // distance to wall computation .
"""

import numpy as np
from mpi4py import MPI

import Converter.Internal as I
import Converter.PyTree   as C

import Geometry        as GEO
import ExtractBoundary as EBD
import Pypdm.Pypdm     as PDM

# --------------------------------------------------------------------------
def computeWallDistance(SkeletonTree, tree, mpicomm,
                        BCTypeToExtract = None,
                        FlowSolutionName='FlowSolution#Init'):
  """
  """
  if(BCTypeToExtract is None):
    findWall        = True
    BCTypeToExtract = []
  else:
    findWall = False

  # I/ Prepare info to build absolute numbering of [Cell] and [Face/Vtx] for surfaces
  #    Utilisation de ParaDiGM dans un cas particulier : ipart = 1
  n_cell_t = 0
  FamBase  = I.getNodesFromType2(SkeletonTree, 'Family_t')
  Zones    = I.getNodesFromType2(tree, 'Zone_t')
  for Zone in Zones:
    ZoneType = I.getNodeFromType1(Zone, 'ZoneType_t')
    if(ZoneType[1].tostring() == 'Unstructured'):
      n_cell_t += Zone[1][0,1]
    else:
      dim = I.getZoneDim(Zone)
      n_cell_t += max(dim[1]-1,1)*max(dim[2]-1,1)*max(dim[3]-1,1)
    if(findWall):
      ZoneBCs  = I.getNodeFromType1(Zone, 'ZoneBC_t')
      BCs      = I.getNodesFromType1(ZoneBCs, 'BC_t')
      for BC in BCs:
        bctype = BC[1].tostring()
        if(bctype == 'FamilySpecified'):
          FamN       = I.getNodeFromType1(BC, 'FamilyName_t')
          FamilyName = FamN[1].tostring()
          FamBaseN   = I.getNodeFromName(FamBase, FamilyName)
          FamilyBCN  = I.getNodeFromType1(FamBaseN, 'FamilyBC_t')
          FamType    = FamilyBCN[1].tostring()
          if(FamType in ['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal ']):
            BCTypeToExtract.append(FamilyName)

  # debug = file('debug_{0}'.format(mpicomm.Get_rank()), 'w')
  # debug.write("BCTypeToExtract  : {0}\n".format(BCTypeToExtract))

  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "computeConcatenateCellCenter ..."
  # mpicomm.Barrier()
  CellCenterG = GEO.computeConcatenateCellCenter(tree)
  shiftLNToGN = mpicomm.scan( n_cell_t, op=MPI.SUM) - n_cell_t
  CellLNToGN  = np.linspace(shiftLNToGN+1, n_cell_t+shiftLNToGN, num=n_cell_t, dtype='int32')

  # > Concatenate connectivity
  # Dans PyPdm on ne sert pas de FaceLNToGN/VtxLNToGN
  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "computeConcatenateFace ..."
  # mpicomm.Barrier()
  FaceVtx, FaceVtxIdx, FaceLNToGN, CellFace, CellFaceIdx, VtxCoord, VtxLNToGN = GEO.computeConcatenateFace(tree, mpicomm)

  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "extractSurfMeshBoundary ..."
  # mpicomm.Barrier()
  FaceVtxBnd, FaceVtxBndIdx, FaceLNToGNBnd, VtxBnd, VtxLNToGNBnd = EBD.extractSurfMeshBoundary(tree, mpicomm, BCTypeToExtract)
  nFaceBnd = FaceVtxBndIdx.shape[0]-1
  nVtxBnd  = VtxLNToGNBnd.shape[0]

  if(FaceLNToGNBnd.shape[0] == 0):
    nFaceBndTot = 0
  else:
    nFaceBndTot = np.max(FaceLNToGNBnd)

  if(VtxLNToGNBnd.shape[0] == 0):
    nVtxBndTot = 0
  else:
    nVtxBndTot  = np.max(VtxLNToGNBnd)

  nFaceBndTot = mpicomm.allreduce(nFaceBndTot, op=MPI.MAX)
  nVtxBndTot  = mpicomm.allreduce(nVtxBndTot , op=MPI.MAX)
  # if(mpicomm.Get_rank() == 0):
  #   debug = file('debug_{0}'.format(mpicomm.Get_rank()), 'w')
  #   debug.write("nVtxBndTot  : {0}\n".format(nVtxBndTot))
  #   debug.write("nFaceBndTot : {0}\n".format(nFaceBndTot))
  #   debug.close()

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # II/ Exchange info in order to build LNToGn properly

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # III/ Build the array to pass to ParaDiGM
  # CenterCell
  # SurfConnectivity
  n_part = 1
  i_part = 0
  dist = PDM.DistCellCenterSurf(mpicomm)
  n_face_t = FaceVtxIdx.shape[0]-1
  n_vtx_t  = VtxCoord.shape[0]/3

  # > Panic debug
  # debug = file('debugIn_{0}'.format(mpicomm.Get_rank()), 'w')
  # debug.write("n_cell_t    : {0}\n".format(n_cell_t))
  # debug.write("n_face_t    : {0}\n".format(n_face_t))
  # debug.write("nVtxBndTot  : {0}\n".format(nVtxBndTot))
  # debug.write("nFaceBndTot : {0}\n".format(nFaceBndTot))
  # debug.write("n_vtx_t     : {0}\n".format(n_vtx_t))
  # debug.write("CellFaceIdx : {0}\n".format(CellFaceIdx))
  # debug.write("CellFace    : {0}\n".format(CellFace))
  # debug.write("CellCenterG : {0}\n".format(CellCenterG))
  # debug.write("CellLNToGN  : {0}\n".format(CellLNToGN))
  # debug.write("FaceVtxIdx  : {0}\n".format(FaceVtxIdx))
  # debug.write("FaceVtx     : {0}\n".format(FaceVtx))
  # debug.write("FaceLNToGN  : {0}\n".format(FaceLNToGN))
  # debug.write("VtxCoord    : {0}\n".format(VtxCoord))
  # debug.write("VtxLNToGN   : {0}\n".format(VtxLNToGN))

  # debug.write("FaceVtxBndIdx  : {0}\n".format(FaceVtxBndIdx))
  # debug.write("FaceVtxBnd     : {0}\n".format(FaceVtxBnd))
  # debug.write("FaceLNToGNBnd  : {0}\n".format(FaceLNToGNBnd))
  # debug.write("nVtxBnd        : {0}\n".format(nVtxBnd))
  # debug.write("VtxBnd         : {0}\n".format(VtxBnd))
  # debug.write("VtxLNToGNBnd   : {0}\n".format(VtxLNToGNBnd))
  # print "*"*1000
  # return

  # > Attention en // ici Totaal Total maybe ?
  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "vol_mesh_global_data_set ..."
  # mpicomm.Barrier()
  dist.vol_mesh_global_data_set(n_cell_t, n_face_t, n_vtx_t, n_part)

  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "vol_mesh_part_set ..."
  # mpicomm.Barrier()
  dist.vol_mesh_part_set(i_part,
                             n_cell_t,
                             CellFaceIdx,
                             CellFace,
                             CellCenterG,
                             CellLNToGN,
                             n_face_t,
                             FaceVtxIdx,
                             FaceVtx,
                             FaceLNToGN,
                             n_vtx_t,
                             VtxCoord,
                             VtxLNToGN)

  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "surf_mesh_global_data_set ..."
  # mpicomm.Barrier()
  dist.surf_mesh_global_data_set(nFaceBndTot, nVtxBndTot, n_part)

  # mpicomm.Barrier()
  # if(mpicomm.Get_rank() == 0):
  #   print "surf_mesh_part_set ..."
  # mpicomm.Barrier()
  dist.surf_mesh_part_set(0, nFaceBnd,
                                 FaceVtxBndIdx,
                                 FaceVtxBnd,
                                 FaceLNToGNBnd,
                                 nVtxBnd,
                                 VtxBnd,
                                 VtxLNToGNBnd)

  # V/ Compute !
  dist.compute()

  # VI/ Post
  walldist = dist.get(0, 0)
