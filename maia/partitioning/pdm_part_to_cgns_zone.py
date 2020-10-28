import Converter.Internal as I
import numpy              as NPY

from maia.connectivity import connectivity_transform as CNT
from .bnd_pdm_to_cgns  import bnd_pdm_to_cgns
from .zgc_pdm_to_cgns  import zgc_created_pdm_to_cgns, zgc_original_pdm_to_cgns

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

def pdm_vtx_to_cgns_grid_coordinates(zone, dims, data):
  """
  """
  grid_c = I.newGridCoordinates(parent=zone)
  print(data.keys())
  I.newDataArray('CoordinateX', data['npVertex'][0::3], parent=grid_c)
  I.newDataArray('CoordinateY', data['npVertex'][1::3], parent=grid_c)
  I.newDataArray('CoordinateZ', data['npVertex'][2::3], parent=grid_c)

def pdm_elmt_to_cgns_elmt(zone, dims, data):
  """
  """
  pdm_face_cell = data['npFaceCell']
  pe            = NPY.empty((pdm_face_cell.shape[0]//2, 2), dtype=pdm_face_cell.dtype, order='F')
  CNT.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)

  ngon_n = I.createUniqueChild(zone, 'NGonElements', 'Elements_t', value=[22,0])
  I.newDataArray('ElementConnectivity', data['npFaceVertex']   , parent=ngon_n)
  I.newDataArray('ElementStartOffset' , data['npFaceVertexIdx'], parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                     , parent=ngon_n)
  I.createNode('ElementRange', 'IndexRange_t',
               [1, dims['nFace']], parent=ngon_n)

  nface_n = I.createUniqueChild(zone, 'NFacElements', 'Elements_t', value=[23,0])
  I.newDataArray('ElementConnectivity', data['npCellFace']   , parent=nface_n)
  I.newDataArray('ElementStartOffset' , data['npCellFaceIdx'], parent=nface_n)
  I.createNode('ElementRange', 'IndexRange_t',
               [dims['nFace']+1, dims['nFace']+dims['nCell']+1], parent=nface_n)



def pdm_part_to_cgns_zone(zone, dist_zone, dims, data, comm):
  """
  """
  save_in_tree_part_info(zone, dims, data, comm)
  pdm_vtx_to_cgns_grid_coordinates(zone, dims, data)
  pdm_elmt_to_cgns_elmt(zone, dims, data)

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

  bnd_pdm_to_cgns(zone, dist_zone, comm)
  zgc_original_pdm_to_cgns(zone, dist_zone, comm)

  # zgc_created_pdm_to_cgns(zone, dist_zone[0], comm)
