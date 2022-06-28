import numpy              as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import layouts

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

      cur_rank, cur_part = MT.conv.get_part_suffix(I.getName(p_zone))
      gcname = MT.conv.name_intra_gc(cur_rank, cur_part, opp_rank, opp_part)
      join_n = I.newGridConnectivity(name      = gcname,
                                     donorName = MT.conv.add_part_suffix(I.getName(d_zone), opp_rank, opp_part),
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

def pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data, connectivity_as="Element"):
  """
  """
  ngon_zone = [e for e in I.getNodesFromType1(d_zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n'] != []
  if  ngon_zone or connectivity_as == 'NGon':
    n_face        = dims['n_face']
    n_cell        = dims['n_cell']
    pdm_face_cell = data['np_face_cell']
    pe = np.empty((n_face, 2), dtype=pdm_face_cell.dtype, order='F')
    layouts.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)
    pe += n_face * (pe > 0)

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
    MT.newGlobalNumbering({'Element' : data['np_face_ln_to_gn']}, ngon_n)

    nface_n = I.createUniqueChild(p_zone, nface_name, 'Elements_t', value=[23,0])
    I.newDataArray('ElementConnectivity', data['np_cell_face']     , parent=nface_n)
    I.newDataArray('ElementStartOffset' , data['np_cell_face_idx'] , parent=nface_n)
    I.newPointRange('ElementRange'      , [n_face+1, n_face+n_cell], parent=nface_n)
    MT.newGlobalNumbering({'Element' : data['np_cell_ln_to_gn']}, nface_n)

  else:
    elt_section_nodes = I.getNodesFromType1(d_zone, "Elements_t")
    assert len(elt_section_nodes) == len(data['2dsections']) + len(data['3dsections'])

    #elt_section_node can have 2d or 3d elements first
    first_elt_node = elt_section_nodes[0]
    first_elt_dim  = PT.Element.Dimension(first_elt_node)
    if first_elt_dim == 2:
      first_sections, second_sections = data['2dsections'], data['3dsections']
    elif first_elt_dim == 3:
      first_sections, second_sections = data['3dsections'], data['2dsections']
    n_elt_cum = 0
    n_elt_cum_d = 0
    for i_section, section in enumerate(first_sections + second_sections):
      if i_section == len(first_sections): #Reset the dimension shift when changing dim
        n_elt_cum_d = 0
      elt = elt_section_nodes[i_section]
      n_i_elt = section['np_connec'].size // PT.Element.NVtx(elt)
      elt_n = I.createUniqueChild(p_zone, I.getName(elt), 'Elements_t', value=I.getValue(elt))
      I.newDataArray('ElementConnectivity', section['np_connec']       , parent=elt_n)
      I.newPointRange('ElementRange'      , [n_elt_cum+1, n_elt_cum+n_i_elt], parent=elt_n)
      numberings = {
          # Original position in the section,
          'Element' : section['np_numabs'] - n_elt_cum_d,
          # Original position in the concatenated sections of same dimension
          'Sections' : section['np_numabs']
          }
      #Original position in the numbering of all elements of the dim: for example, for faces, this is
      # the gnum in the description of all the faces (and not only faces describred in sections
      if section['np_parent_entity_g_num'] is not None:
        numberings['ImplicitEntity'] = section['np_parent_entity_g_num']
      # Corresponding face in the array of all faces described by a section,
      # after face renumbering
      # Local number of entity in the reordered cells of the partition
      # (for faces, note that only face explicitly described are renumbered)
      lnum_node = I.createNode(':CGNS#LocalNumbering', 'UserDefinedData_t', parent=elt_n)
      I.newDataArray('ExplicitEntity', section['np_parent_num'], parent=lnum_node)

      MT.newGlobalNumbering(numberings, elt_n)

      n_elt_cum   += n_i_elt
      n_elt_cum_d += PT.Element.Size(elt_section_nodes[i_section])

def pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm, options):
  """
  """
  #Dims and data should be related to the dist zone and of size n_parts
  part_zones = list()
  for i_part, (dims, data) in enumerate(zip(l_dims, l_data)):

    part_zone = I.newZone(name  = MT.conv.add_part_suffix(I.getName(dist_zone), comm.Get_rank(), i_part),
                          zsize = [[dims['n_vtx'],dims['n_cell'],0]],
                          ztype = 'Unstructured')

    if options['dump_pdm_output']:
      dump_pdm_output(part_zone, dims, data)
    pdm_vtx_to_cgns_grid_coordinates(part_zone, dims, data)
    pdm_elmt_to_cgns_elmt(part_zone, dist_zone, dims, data, options['output_connectivity'])

    output_loc = options['part_interface_loc']
    zgc_name = 'ZoneGridConnectivity#Vertex' if output_loc == 'Vertex' else 'ZoneGridConnectivity'
    zgc_created_pdm_to_cgns(part_zone, dist_zone, dims, data, output_loc, zgc_name)

    lngn_zone = MT.newGlobalNumbering(parent=part_zone)
    I.newDataArray('Vertex', data['np_vtx_ln_to_gn'], parent=lngn_zone)
    I.newDataArray('Cell', data['np_cell_ln_to_gn'], parent=lngn_zone)

    part_zones.append(part_zone)

  return part_zones