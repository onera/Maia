import Converter.Internal as I
import maia.sids.Internal_ext as IE
import numpy              as np

from maia.sids import sids
from maia.sids import conventions as conv
from maia.connectivity import connectivity_transform as CNT

def dump_pdm_output(p_zone, dims, data):
  """
  Write PDM output in part_tree (for debug)
  """
  ppart_node = I.createUniqueChild(p_zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for dim_key, dim_val in dims.items():
    I.newDataArray(dim_key, dim_val, parent=ppart_node)
  for data_key, data_val in data.items():
    if isinstance(data_val, np.ndarray):
      I.newDataArray(data_key, np.copy(data_val), parent=ppart_node)

def zgc_created_pdm_to_cgns(p_zone, d_zone, dims, data, grid_loc='FaceCenter', zgc_name='ZoneGridConnectivity'):
  """
  Create by splitting
  """
  if grid_loc not in ['FaceCenter', 'Vertex']:
    raise NotImplementedError("Unvalid specified entity")
  entity = 'face' if grid_loc == 'FaceCenter' else 'vtx'

  entity_part_bound_proc_idx = data['np_{0}_part_bound_proc_idx'.format(entity)]
  entity_part_bound_part_idx = data['np_{0}_part_bound_part_idx'.format(entity)]
  entity_part_bound_tmp      = data['np_{0}_part_bound'         .format(entity)]

  entity_part_bound = entity_part_bound_tmp.reshape((4, entity_part_bound_tmp.shape[0]//4), order='F')
  entity_part_bound = entity_part_bound.transpose()

  zgc_n = I.newZoneGridConnectivity(name=zgc_name, parent=p_zone)

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

      opp_rank = entity_part_bound[beg_pl, 1]
      opp_part = entity_part_bound[beg_pl, 2]-1

      cur_rank, cur_part = conv.get_part_suffix(I.getName(p_zone))
      gcname = conv.name_intra_gc(cur_rank, cur_part, opp_rank, opp_part)
      join_n = I.newGridConnectivity(name      = gcname,
                                     donorName = conv.add_part_suffix(I.getName(d_zone), opp_rank, opp_part),
                                     ctype     = 'Abutting1to1',
                                     parent    = zgc_n)

      I.newGridLocation(grid_loc, parent=join_n)
      I.newPointList(name='PointList'     , value=pl , parent=join_n)
      I.newPointList(name='PointListDonor', value=pld, parent=join_n)


def pdm_vtx_to_cgns_grid_coordinates(p_zone, dims, data):
  """
  """
  grid_c = I.newGridCoordinates(parent=p_zone)
  I.newDataArray('CoordinateX', data['np_vtx_coord'][0::3], parent=grid_c)
  I.newDataArray('CoordinateY', data['np_vtx_coord'][1::3], parent=grid_c)
  I.newDataArray('CoordinateZ', data['np_vtx_coord'][2::3], parent=grid_c)

def pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data):
  """
  """
  if (ngon_zone):
  ngon_zone = [e for e in I.getNodesFromType1(d_zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n'] != []
    n_face        = dims['n_face']
    n_cell        = dims['n_cell']
    pdm_face_cell = data['np_face_cell']
    pe = np.empty((n_face, 2), dtype=pdm_face_cell.dtype, order='F')
    CNT.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)

    ngon_name  = 'NGonElements'
    nface_name = 'NFaceElements'
    for elt in I.getNodesFromType1(d_zone, 'Elements_t'):
      if I.getValue(elt)[0] == 22:
        ngon_name = I.getName(elt)
      elif I.getValue(elt)[0] == 23:
        nface_name = I.getName(elt)

    ngon_n = I.createUniqueChild(p_zone, ngon_name, 'Elements_t', value=[22,0])
    I.newDataArray('ElementConnectivity', data['np_face_vtx']    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , data['np_face_vtx_idx'], parent=ngon_n)
    I.newDataArray('ParentElements'     , pe                     , parent=ngon_n)
    I.newPointRange('ElementRange'      , [1, n_face]            , parent=ngon_n)
    IE.newGlobalNumbering({'Element' : data['np_face_ln_to_gn']}, ngon_n)

    nface_n = I.createUniqueChild(p_zone, nface_name, 'Elements_t', value=[23,0])
    I.newDataArray('ElementConnectivity', data['np_cell_face']     , parent=nface_n)
    I.newDataArray('ElementStartOffset' , data['np_cell_face_idx'] , parent=nface_n)
    I.newPointRange('ElementRange'      , [n_face+1, n_face+n_cell], parent=nface_n)
    IE.newGlobalNumbering({'Element' : data['np_cell_ln_to_gn']}, nface_n)

  else:
    elt_section_nodes = I.getNodesFromType(d_zone, "Elements_t")
    assert len(elt_section_nodes) == len(data['2dsections']) + len(data['3dsections'])

    #elt_section_node should have 2d first, then 3d
    n_elt_cum = 0
    for i_section, section in enumerate(data['2dsections'] + data['3dsections']):
      elt = elt_section_nodes[i_section]
      n_i_elt = section['np_connec'].size // sids.ElementNVtx(elt)
      elt_n = I.createUniqueChild(p_zone, I.getName(elt), 'Elements_t', value=I.getValue(elt))
      I.newDataArray('ElementConnectivity', section['np_connec']       , parent=elt_n)
      I.newPointRange('ElementRange'      , [n_elt_cum+1, n_elt_cum+n_i_elt], parent=elt_n)
      n_elt_cum += n_i_elt
      IE.newGlobalNumbering({'Element' : section['np_numabs']}, elt_n)

def pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm, options):
  """
  """
  #Dims and data should be related to the dist zone and of size n_parts
  part_zones = list()
  for i_part, (dims, data) in enumerate(zip(l_dims, l_data)):

    part_zone = I.newZone(name  = conv.add_part_suffix(I.getName(dist_zone), comm.Get_rank(), i_part),
                          zsize = [[dims['n_vtx'],dims['n_cell'],0]],
                          ztype = 'Unstructured')

    if options['dump_pdm_output']:
      dump_pdm_output(part_zone, dims, data)
    pdm_vtx_to_cgns_grid_coordinates(part_zone, dims, data)
    pdm_elmt_to_cgns_elmt(part_zone, dist_zone, dims, data)

    output_loc = options['part_interface_loc']
    zgc_name = 'ZoneGridConnectivity#Vertex' if output_loc == 'Vertex' else 'ZoneGridConnectivity'
    zgc_created_pdm_to_cgns(part_zone, dist_zone, dims, data, output_loc, zgc_name)

    lngn_zone = IE.newGlobalNumbering(parent=part_zone)
    I.newDataArray('Vertex', data['np_vtx_ln_to_gn'], parent=lngn_zone)
    I.newDataArray('Cell', data['np_cell_ln_to_gn'], parent=lngn_zone)

    part_zones.append(part_zone)

  return part_zones
