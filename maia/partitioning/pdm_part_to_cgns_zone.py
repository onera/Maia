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

  ppart_node = I.createUniqueChild(zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for k in dims.keys():
    I.newDataArray(k, dims[k], parent=ppart_node)

  for k in data.keys():
    if k!="np_elt_vtx_idx" and k!="np_elt_vtx" : #type(data[k]) is not list
      I.newDataArray(k, NPY.copy(data[k]), parent=ppart_node)

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
    pe            = NPY.empty((pdm_face_cell.shape[0]//2, 2), dtype=pdm_face_cell.dtype, order='F')
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




def pdm_part_to_cgns_zone(zone, dist_zone, dims, data, comm):
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

  # TODO
  #bnd_pdm_to_cgns(zone, dist_zone, comm)
  #zgc_original_pdm_to_cgns(zone, dist_zone, comm)

  #zgc_created_pdm_to_cgns(zone, dist_zone, comm, 'face')
  zgc_created_pdm_to_cgns(zone, dist_zone, comm, 'vtx', 'ZoneGridConnectivity#Vertex')
