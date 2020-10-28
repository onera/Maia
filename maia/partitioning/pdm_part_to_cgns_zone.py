import Converter.Internal as I
import numpy              as NPY

def save_in_tree_part_info(zone, dims, data, comm):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  child = I.createUniqueChild(zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for k in dims.keys():
    I.newDataArray(k, dims[k], parent=child)

  for k in data.keys():
    if(data[k] is not None):
      I.newDataArray(k, NPY.copy(data[k]), parent=child)

  I.newDataArray('iproc', i_rank, parent=child)



def pdm_part_to_cgns_zone(zone, dims, data, comm):
  """
  """
  save_in_tree_part_info(zone, dims, data, comm)

  # LYT._convertVtxAndFaceCellForCGNS(zone)
  # if I.getNodeFromName2(dist_zone, 'ElementStartOffset') is not None:
  #   NGonNode = I.getNodeFromName1(zone, 'NGonElements')
  #   EC  = I.getNodeFromName(child, 'npFaceVertex')[1]
  #   ESO = I.getNodeFromName(child, 'npFaceVertexIdx')[1]
  #   I.newDataArray('ElementConnectivity', EC, parent=NGonNode)
  #   I.newDataArray('ElementStartOffset', ESO, parent=NGonNode)
  #   I.createNode('ElementRange', 'IndexRange_t',
  #                [1, dims['nFace']], parent=NGonNode)
  # else:
  #   TBX._convertMorseToNGon2(zone)
  #   TBX._convertMorseToNFace(zone)

  # PdmBoundariesToCgns(zone, dist_zone, Comm)
  # PdmOriginalJoinsToCgns(zone, dist_zone, Comm)

  # PdmCreatedJoinsToCgns(zone, dist_zone[0], Comm)
