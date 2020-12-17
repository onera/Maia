import logging     as LOG
import numpy       as np

import     mpi4py.MPI        as MPI

import Converter.PyTree   as C
import Converter.Internal as I

import Pypdm.Pypdm as PDM

# --------------------------------------------------------------------------
def distFlowSolToPartFlowSol(dist_tree, part_tree, dZoneToPart, comm):
  """
  Transfert all the flowSolution nodes found in dist_tree to the
  corresponding partitions in part_tree.

  Args:
      dist_tree (pyTree)  : A distributed pyTree
      part_tree (pyTree)  : A partitioned pyTree
      dZoneToPart (dict) : Mapping from DistZones to partZones : for each
                           distzone name (key), list of size partN on this
                           zone for this process (value). The content of the
                           list is unused here, only size matters.
      comm (MPI.Comm)    : MPI Communicator (from mpi4py)
  """

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info("### TransfertTreeData::distFlowSolToPartFlowSol")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  iRank = comm.Get_rank()
  nRank = comm.Get_size()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Do a blockToPart on each zone, for each FlowSolution
  for distZoneName in dZoneToPart:
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Get FlowSolution nodes and Distribution node
    distZone          = I.getNodeFromName2(dist_tree, distZoneName)
    FlowSolutionNodes = I.getNodesFromType1(distZone, "FlowSolution_t")
    DistribUD         = I.getNodeFromName1(distZone, ':CGNS#Distribution')
    nParts  = len(dZoneToPart[distZoneName])
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Verbose
    LOG.debug("   Treat initial zone {0} : partN is {1}".format(
      distZoneName, nParts))
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    for FlowSolutionNode in FlowSolutionNodes:
      # --------------------------------------------------------------------
      # > Initialize
      pLNToGN = list()
      dData   = dict()
      pData   = dict()
      LOG.debug("     Treat flowsolution node {0}".format(FlowSolutionNode[0]))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Get grid location to use the good LNToGN
      GridLocNode = I.getNodeFromType1(FlowSolutionNode, 'GridLocation_t')
      GridLoc     = I.getValue(GridLocNode)

      if(GridLoc == 'CellCenter'):
        PpartNodeName = 'np_cell_ln_to_gn'
        DistriName    = 'Distribution_cell'
        npElemName    = 'nCell'
      elif(GridLoc == 'Vertex'):
        PpartNodeName = 'np_vtx_ln_to_gn'
        DistriName    = 'Distribution_vtx'
        npElemName    = 'nVertex'
      elif(GridLoc == 'FaceCenter'):
        PpartNodeName = 'np_face_ln_to_gn'
        DistriName    = 'Distribution_face'
        npElemName    = 'nFace'
      else:
        LOG.error(' '*6 + 'Bad grid location for solution {0} on zone {1}'.format(
          FlowSolutionNode[0], distZoneName))
        comm.abort()
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Get distributed data and init partData dict
      for DataArray in I.getNodesFromType1(FlowSolutionNode, 'DataArray_t'):
        dData[DataArray[0]] = DataArray[1]
        pData[DataArray[0]] = list()
      # LOG.debug("     dData : {0}".format(dData))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > (Re)compute distribution because it's not stored
      DistributionSol = I.getNodeFromName1(DistribUD, DistriName)[1]
      ndElem = DistributionSol[1] - DistributionSol[0]
      PDMDistribution = np.empty((nRank + 1), order='C', dtype='int32')
      PDMDistribution[0]  = 0
      PDMDistribution[1:] = comm.allgather(ndElem)
      for j in range(nRank):
        PDMDistribution[j+1] = PDMDistribution[j+1] + PDMDistribution[j]
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Loop over parts to get LNToGN
      for iPart in range(nParts):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get zone
        partZoneName = "{0}.P{1}.N{2}".format(distZoneName, iRank, iPart)
        partZone = I.getNodeFromName2(part_tree, partZoneName)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get LNToGN
        PpartNode = I.getNodeFromName1(partZone, ':CGNS#Ppart')
        LNToGNNPY = I.getNodeFromName1(PpartNode, PpartNodeName)[1]
        pLNToGN.append(LNToGNNPY)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Initialize FlowSolution in the partZone - mandatory for PDM
        PartFlowSol = I.newFlowSolution(name=FlowSolutionNode[0],
                                        gridLocation=GridLoc, parent=partZone)
        npElem = I.getNodeFromName1(PpartNode, npElemName)[1][0]
        for field in dData:
          npyType = dData[field].dtype
          emptyArray = np.empty(npElem, order='C', dtype=npyType)
          I.createChild(PartFlowSol, field, 'DataArray_t', emptyArray)
          pData[field].append(emptyArray)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Move block data to part data
      BTP = PDM.BlockToPart(PDMDistribution, comm, pLNToGN, nParts)
      BTP.BlockToPart_Exchange(dData, pData)
      # LOG.debug("     pData : {0}".format(pData))
      # --------------------------------------------------------------------

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# --------------------------------------------------------------------------
def distDataSetToPartDataSet(dist_tree, part_tree, dZoneToPart, comm):
  """
  Transfert all the BCDataSet nodes found in dist_tree to the
  corresponding partitions in part_tree.

  Args:
      dist_tree (pyTree)  : A distributed pyTree
      part_tree (pyTree)  : A partitioned pyTree
      dZoneToPart (dict) : Mapping from DistZones to partZones : for each
                           distzone name (key), list of size partN on this
                           zone for this process (value). The content of the
                           list is unused here, only size matters.
      comm (MPI.Comm)    : MPI Communicator (from mpi4py)
  """

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info("### TransfertTreeData::distDataSetToPartDataSet")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  iRank = comm.Get_rank()
  nRank = comm.Get_size()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  BaseName = I.getNodeFromType1(part_tree, 'CGNSBase_t')[0]
  ZonesPathsNode = I.getNodeFromPath(part_tree, '/'+BaseName+'/:Ppart#ZonePaths')
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Do a blockToPart on each zone, for each BCDataSet
  for distZoneName in dZoneToPart:
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Get FlowSolution nodes and Distribution node
    distZone          = I.getNodeFromName2(dist_tree, distZoneName)
    distZoneBC        = I.getNodeFromType1(distZone, "ZoneBC_t")
    zonePathsNode     = I.getNodeFromName1(ZonesPathsNode, distZoneName)
    nParts  = len(dZoneToPart[distZoneName])
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Verbose
    LOG.debug("   Treat initial zone {0} : partN is {1}".format(
      distZoneName, nParts))
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    for BC in I.getNodesFromType1(distZoneBC, "BC_t"):
      # --------------------------------------------------------------------
      # > Initialize
      pLNToGN = list()
      dData   = dict()
      pData   = dict()
      LOG.debug("     Treat BC node {0}".format(BC[0]))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Get distributed data and init partData dict
      BCDataSetValue = dict()
      zonePathBCName = '##'.join(['BC', distZoneBC[0], BC[0]])
      zonePathBCNode = I.getNodeFromName1(zonePathsNode, zonePathBCName)
      for BCDataSet in I.getNodesFromType1(BC, "BCDataSet_t"):
        BCDataSetValue[BCDataSet[0]] = I.getValue(BCDataSet)
        for BCData in I.getNodesFromType1(BCDataSet, "BCData_t"):
          pathNodeName = '##'.join([BCDataSet[0], BCData[0]])
          anyArray = I.getNodeFromType1(BCData, 'DataArray_t')[1]
          pathNodeValue = I.getValue(BCDataSet) + '##' + str(anyArray.dtype)
          for DataArray in I.getNodesFromType1(BCData, "DataArray_t"):
            dData['/'.join([BCDataSet[0], BCData[0], DataArray[0]])] = DataArray[1]
            pData['/'.join([BCDataSet[0], BCData[0], DataArray[0]])] = list()
            pathNodeValue += '##' + DataArray[0]
        I.createNode(pathNodeName, 'UserDefinedData_t',
                     pathNodeValue, parent=zonePathBCNode)
      # LOG.debug("     dData : {0}".format(dData))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > (Re)compute distribution because it's not stored
      DistriBC = I.getNodeFromPath(BC, ':CGNS#Distribution/DistributionBnd')[1]
      ndElem = DistriBC[1] - DistriBC[0]
      PDMDistribution = np.empty((nRank + 1), order='C', dtype='int32')
      PDMDistribution[0]  = 0
      PDMDistribution[1:] = comm.allgather(ndElem)
      for j in range(nRank):
        PDMDistribution[j+1] = PDMDistribution[j+1] + PDMDistribution[j]
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Loop over parts to get LNToGN
      for iPart in range(nParts):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get zone
        partZoneName = "{0}.P{1}.N{2}".format(distZoneName, iRank, iPart)
        partZone = I.getNodeFromName2(part_tree, partZoneName)
        partBCName = distZoneBC[0] + "/{0}.P{1}.N{2}".format(BC[0], iRank, iPart)
        partBC     = I.getNodeFromPath(partZone, partBCName)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get LNToGN
        if partBC is not None:
          LNToGNNPY = I.getNodeFromName1(partBC, "LNtoGN")[1]
        else:
          LNToGNNPY = np.empty(0, dtype=np.int32)
        pLNToGN.append(LNToGNNPY)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Initialize part arrays - mandatory for PDM
        # > Also put reference if parttree if BC exists
        for field, array in dData.items():
          emptyArray = np.empty([1,len(LNToGNNPY)], order='C', dtype=array.dtype)
          pData[field].append(emptyArray)
          if len(LNToGNNPY) > 0:
            dataSetName, dataName, arrayName = field.split('/')
            dataSetN = I.getNodeFromName1(partBC, dataSetName)
            if dataSetN is None:
              dataSetValue = BCDataSetValue[dataSetName]
              dataSetN = I.newBCDataSet(dataSetName, dataSetValue, parent=partBC)
            dataN = I.getNodeFromName1(dataSetN, dataName)
            if dataN is None:
              dataN = I.newBCData(dataName, parent=dataSetN)
            I.createChild(dataN, arrayName, 'DataArray_t', emptyArray)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Move block data to part data
      BTP = PDM.BlockToPart(PDMDistribution, comm, pLNToGN, nParts)
      BTP.BlockToPart_Exchange(dData, pData)
      # LOG.debug("     pData : {0}".format(pData))
      # --------------------------------------------------------------------
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# --------------------------------------------------------------------------
def pFlowSolution_to_dFlowSolution(dist_tree, part_tree, dZoneToPart, comm):
  """
  Transfert all the flowSolution nodes found in part_tree to the
  corresponding distZones in dist_tree.
  Any flowSolution node pre existing in dist_tree will be removed.

  Args:
      dist_tree (pyTree)  : A distributed pyTree
      part_tree (pyTree)  : A partitioned pyTree
      dZoneToPart (dict) : Mapping from DistZones to partZones : for each
                           distzone name (key), list of size partN on this
                           zone for this process (value). The content of the
                           list is unused here, only size matters.
      comm (MPI.Comm)    : MPI Communicator (from mpi4py)
  """
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info("### TransfertTreeData::pFlowSolution_to_dFlowSolution")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  iRank = comm.Get_rank()
  nRank = comm.Get_size()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Init flow solution names, grid location and datatype. This is usefull
  #   a. to declare only one PTB structure per different gridlocation
  #   b. to declare empty data (for PDM) if proc have no partition on a zone
  #  We assume that flowsolution containers have the same name/dataarray on
  #  each zone, and take names from a random partitioned zone.
  #  Troubles can occur if a proc does not have partitions at all
  gridLocToLNGNName = {'CellCenter' : 'np_cell_ln_to_gn',
                       'Vertex'     : 'np_vtx_ln_to_gn',
                       'FaceCenter' : 'np_face_ln_to_gn'}
  gridLocToDistriUD = {'CellCenter' : 'Cell',
                       'Vertex'     : 'Vertex',
                       'FaceCenter' : 'Face'}
  gridLocToVoid = dict()

  partZone = I.getNodeFromType2(part_tree, "Zone_t")
  LOG.debug(' '*4 + "Uses part zone {0} to get list of flow solutions".format(
      partZone[0]))
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  for FlowSolutionNode in I.getNodesFromType1(partZone, "FlowSolution_t"):
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Get grid location to use the good LNToGN
    GridLocNode = I.getNodeFromType1(FlowSolutionNode, 'GridLocation_t')
    GridLoc     = I.getValue(GridLocNode)
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Check and init dict
    if (GridLoc not in gridLocToLNGNName):
      LOG.error(' '*6 + 'Bad grid location for solution {0} : {1} '.format(
        FlowSolutionNode[0], GridLoc))
      comm.Abort()
    if (GridLoc not in gridLocToVoid):
      gridLocToVoid[GridLoc] = dict()
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Store name of FlowSolutions, DataArrays and dataType depending on
    #   the gridLocation
    for DataArray in I.getNodesFromType1(FlowSolutionNode, 'DataArray_t'):
      fieldName = FlowSolutionNode[0] + "/" + DataArray[0]
      npy_type  = DataArray[1].dtype
      gridLocToVoid[GridLoc][fieldName] = [np.empty(0, dtype=npy_type)]
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Verbose
  # LOG.debug(' '*4 + "Found locations and fields :")
  # for loc, dicfields in gridLocToVoid.items():
    # LOG.debug(' '*6 + "{0}".format(loc))
    # for field, data in dicfields.items():
      # LOG.debug(' '*8 + "{0} --> {1}".format(field, data))
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Now trully loop on zone, and partToBlock each gridlocation
  for distZoneName in dZoneToPart:
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Initialize for multiple locations
    nParts  = len(dZoneToPart[distZoneName])
    LOG.debug("  Treat initial zone {0} : partN is {1}".format(
      distZoneName, nParts))

    gridLocToLNToGN = {gLoc : list() for gLoc in gridLocToVoid.keys()}
    gridLocToFields = {gLoc : dict() for gLoc in gridLocToVoid.keys()}
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Get void dict if proc have no part (mandatory for PDM)
    #   Otherwise, we have to erase the empty array to fill with real data
    if (nParts == 0):
      gridLocToFields = gridLocToVoid
    else:
      for gLoc, fields in gridLocToVoid.items():
        gridLocToFields[gLoc] = {field: list() for field in fields}
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Loop over parts (if existing) and fill lntogn and arrays
    for iPart in range(nParts):
      # --------------------------------------------------------------------
      partZoneName = "{0}.P{1}.N{2}".format(distZoneName, iRank, iPart)
      partZone = I.getNodeFromName2(part_tree, partZoneName)
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Get LNToGN
      PpartNode = I.getNodeFromName1(partZone, ':CGNS#Ppart')
      for gridLoc in gridLocToFields:
        LNToGNName = gridLocToLNGNName[gridLoc]
        LNToGNNPY = I.getNodeFromName1(PpartNode, LNToGNName)[1]
        gridLocToLNToGN[gridLoc].append(LNToGNNPY)
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Get data
      FlowSolutionNodes = I.getNodesFromType1(partZone, "FlowSolution_t")
      for FlowSolutionNode in FlowSolutionNodes:
        # LOG.debug("    Found flow solution node '{0}'".format(
          # FlowSolutionNode[0]))
        GridLocNode = I.getNodeFromType1(FlowSolutionNode, 'GridLocation_t')
        gridLoc     = I.getValue(GridLocNode)
        for DataArray in I.getNodesFromType1(FlowSolutionNode, 'DataArray_t'):
          fieldName = FlowSolutionNode[0] + "/" + DataArray[0]
          gridLocToFields[gridLoc][fieldName].append(DataArray[1])
      # --------------------------------------------------------------------
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Remove pre existing flow solution nodes from disttree
    distZone = I.getNodeFromName2(dist_tree, distZoneName)
    I._rmNodesByType1(distZone, "FlowSolution_t")
    DistribUD = I.getNodeFromName1(distZone, ':CGNS#Distribution')
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Create a PTB for each kind of location
    for gridLoc in gridLocToFields:
      # --------------------------------------------------------------------
      pLNToGN = gridLocToLNToGN[gridLoc]
      pData   = gridLocToFields[gridLoc]
      dData   = dict()
      pWeight = None
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > (Re)compute distribution because it's not stored
      distriName      = gridLocToDistriUD[gridLoc]
      DistributionSol = I.getNodeFromName1(DistribUD, distriName)[1]
      ndElem = DistributionSol[1] - DistributionSol[0]
      PDMDistribution = np.empty((nRank + 1), order='C', dtype='int32')
      PDMDistribution[0]  = 0
      PDMDistribution[1:] = comm.allgather(ndElem)
      for j in range(nRank):
        PDMDistribution[j+1] = PDMDistribution[j+1] + PDMDistribution[j]
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      PTB = PDM.PartToBlock(comm, pLNToGN, pWeight, nParts,
                            t_distrib = 0,
                            t_post    = 1,
                            t_stride  = 0,
                            userDistribution = PDMDistribution)
      LOG.debug(' '*4 +"Exchange solutions located at {0}".format(gridLoc))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      #LOG.debug(' '*4 + "partdata : {0}".format(pData))
      PTB.PartToBlock_Exchange(dData, pData)
      #LOG.debug(' '*4 + "distdata : {0}".format(dData))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Add received data in dist_tree
      flowSolutionNames = []
      for path in dData:
        if path.split('/')[0] not in flowSolutionNames:
          flowSolutionNames.append(path.split('/')[0])

      for flowSolutionName in flowSolutionNames:
        distFlowSol = I.newFlowSolution(name=flowSolutionName,
                                        gridLocation=gridLoc,
                                        parent=distZone)

        fields = [path.split('/')[1] for path in dData if
            path.split('/')[0]==flowSolutionName]
        for field in fields:
          I.newDataArray(name=field,
                         value=dData[flowSolutionName + '/' + field],
                         parent=distFlowSol)
      # --------------------------------------------------------------------
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# --------------------------------------------------------------------------
def partDataSetToDistDataSet(dist_tree, part_tree, dZoneToPart, comm):
  """
  Transfert all the BCDataSet nodes found in part_tree to the
  corresponding distZones in dist_tree.
  Any BCDatase node pre existing in dist_tree will be removed.

  Args:
      dist_tree (pyTree)  : A distributed pyTree
      part_tree (pyTree)  : A partitioned pyTree
      dZoneToPart (dict) : Mapping from DistZones to partZones : for each
                           distzone name (key), list of size partN on this
                           zone for this process (value). The content of the
                           list is unused here, only size matters.
      comm (MPI.Comm)    : MPI Communicator (from mpi4py)
  """
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info("### TransfertTreeData::partDataSetToDistDataSet")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  iRank = comm.Get_rank()
  nRank = comm.Get_size()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  BaseName = I.getNodeFromType1(part_tree, 'CGNSBase_t')[0]
  ZonePathsNode = I.getNodeFromPath(part_tree, '/'+BaseName+'/:Ppart#ZonePaths')
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Loop over zones
  for distZoneName in dZoneToPart:
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    nParts  = len(dZoneToPart[distZoneName])
    distZone = I.getNodeFromName2(dist_tree, distZoneName)
    LOG.debug("  Treat initial zone {0} : partN is {1}".format(
      distZoneName, nParts))
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # Filter BCs
    BCAndGCNames = I.getNodeFromName1(ZonePathsNode, distZoneName)
    BCs = [BCOrGC for BCOrGC in I.getChildren(BCAndGCNames) \
        if BCOrGC[0][:2] == 'BC']
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # > Loop over BCs
    for BC in BCs:
      BoundaryType, ContainerName, BoundaryName = BC[0].split('##')
      # --------------------------------------------------------------------
      # > Initialize
      pLNToGN = list()
      dData   = dict()
      pData   = dict()
      LOG.debug("     Treat BC node {0}".format(BoundaryName))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      distZoneBC = I.getNodeFromName1(distZone, ContainerName)
      distBC     = I.getNodeFromName1(distZoneBC, BoundaryName)
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > (Re)compute distribution because it's not stored
      DistriBC = I.getNodeFromPath(distBC, ':CGNS#Distribution/DistributionBnd')[1]
      ndElem = DistriBC[1] - DistriBC[0]
      PDMDistribution = np.empty((nRank + 1), order='C', dtype='int32')
      PDMDistribution[0]  = 0
      PDMDistribution[1:] = comm.allgather(ndElem)
      for j in range(nRank):
        PDMDistribution[j+1] = PDMDistribution[j+1] + PDMDistribution[j]
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Create path node for dataset if unexisting (eg if dataset was
      #   added directly on parttree). Should be done before ?
      if len(I.getChildren(BC)) == 0:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Search BCDataSet and create PathNode
        hasBcDataSet = False
        for iPart in range(nParts):
          partBCPath = "{0}/{1}.P{4}.N{5}/{2}/{3}.P{4}.N{5}".format(
              BaseName, distZoneName, ContainerName, BoundaryName, iRank, iPart)
          partBC = I.getNodeFromPath(part_tree, partBCPath)
          pathNodes = list()
          if partBC is not None:
            BCDataSets = I.getNodesFromType1(partBC, 'BCDataSet_t')
            if len(BCDataSets) != 0:
              hasBcDataSet = True
              for BCDataSet in BCDataSets:
                for BCData in I.getNodesFromType1(BCDataSet, 'BCData_t'):
                  pathNodeName = '##'.join([BCDataSet[0], BCData[0]])
                  anyArray = I.getNodeFromType1(BCData, 'DataArray_t')[1]
                  pathNodeValue = I.getValue(BCDataSet) + '##' + str(anyArray.dtype)
                  for DataArray in I.getNodesFromType1(BCData, "DataArray_t"):
                    pathNodeValue += '##' + DataArray[0]
                  pathNodes.append(I.createNode(pathNodeName, 'UserDefinedData_t',
                                   pathNodeValue))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Share pathnode and add it
        gHasBcDataSet = comm.allgather(hasBcDataSet)
        try:
          sender = gHasBcDataSet.index(True)
          pathNodes = comm.bcast(pathNodes, root=sender)
          for pathNode in pathNodes:
            I.addChild(BC, pathNode)
        except ValueError: #No bcdataset
          pass
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Initialize fields
      BCDataSetValue = dict()
      for DataSet in I.getChildren(BC):
        DataSetPath = DataSet[0].replace('##','/')
        fields = I.getValue(DataSet).split('##')[2:]
        DataSetKind = I.getValue(DataSet).split('##')[0]
        BCDataSetValue[DataSetPath.split('/')[0]] = DataSetKind
        for field in fields:
          DataSetFieldPath = DataSetPath+'/'+field
          pData[DataSetFieldPath] = list()
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Loop over parts to get LNToGN
      for iPart in range(nParts):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get zone and partbc
        partZoneName = "{0}.P{1}.N{2}".format(distZoneName, iRank, iPart)
        partZone = I.getNodeFromName2(part_tree, partZoneName)
        partBCName = "{0}/{1}.P{2}.N{3}".format(
            ContainerName, BoundaryName, iRank, iPart)
        partBC     = I.getNodeFromPath(partZone, partBCName)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get LNToGN
        if partBC is not None:
          LNToGNNPY = I.getNodeFromName1(partBC, "LNtoGN")[1]
        else:
          LNToGNNPY = np.empty(0, dtype=np.int32)
        pLNToGN.append(LNToGNNPY)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > Get part data
        for DataSet in I.getChildren(BC):
          DataSetPath = DataSet[0].replace('##','/')
          npydtype = I.getValue(DataSet).split('##')[1]
          fields = I.getValue(DataSet).split('##')[2:]
          for field in fields:
            DataSetFieldPath = DataSetPath+'/'+field
            if partBC is not None:
              DataSetField = I.getNodeFromPath(partBC, DataSetFieldPath)[1]
              pData[DataSetFieldPath].append(DataSetField)
            else:
              emptyArray = np.empty([1,0], order='C', dtype=npydtype)
              pData[DataSetFieldPath].append(emptyArray)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > PartToBlock
      pWeight = None
      PTB = PDM.PartToBlock(comm, pLNToGN, pWeight, nParts,
                            t_distrib = 0,
                            t_post    = 0,
                            t_stride  = 0,
                            userDistribution = PDMDistribution)
      # LOG.debug("     pData : {0}".format(pData))
      PTB.PartToBlock_Exchange(dData, pData)
      # LOG.debug("     dData : {0}".format(dData))
      # --------------------------------------------------------------------

      # --------------------------------------------------------------------
      # > Add received DataSet in tree
      I._rmNodesByType(distBC, "BCDataSet_t")
      for field, data in dData.items():
        dataSetName, dataName, arrayName = field.split('/')
        dataSetN = I.getNodeFromName1(distBC, dataSetName)
        if dataSetN is None:
          dataSetKind = BCDataSetValue[dataSetName]
          dataSetN = I.newBCDataSet(dataSetName, dataSetKind, parent=distBC)
        dataN = I.getNodeFromName1(dataSetN, dataName)
        if dataN is None:
          dataN = I.newBCData(dataName, parent=dataSetN)
        distField = I.createChild(dataN, arrayName, 'DataArray_t', data)
      # --------------------------------------------------------------------
    # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
