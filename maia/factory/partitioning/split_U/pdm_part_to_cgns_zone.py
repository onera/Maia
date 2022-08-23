import numpy              as np
import itertools

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

def pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data, connectivity_as="Element", keep_empty_sections=False):
  """
  """
  ngon_zone = [e for e in PT.iter_children_from_label(d_zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n'] != []
  if  ngon_zone or connectivity_as == 'NGon':
    n_face        = dims['n_face']
    n_cell        = dims['n_cell']
    pdm_face_cell = data['np_face_cell']
    pe = np.empty((n_face, 2), dtype=pdm_face_cell.dtype, order='F')
    layouts.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)
    pe += n_face * (pe > 0)

    ngon_name  = 'NGonElements'
    nface_name = 'NFaceElements'
    for elt in PT.iter_children_from_label(d_zone, 'Elements_t'):
      if PT.Element.CGNSName(elt) == 'NGON_n':
        ngon_name = I.getName(elt)
      elif PT.Element.CGNSName(elt) == 'NFACE_n':
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
    # if vtx are ordered with:
    #   1. unique first
    #   2. then non-unique but owned
    #   3. then non-unique and non-owned ("ghost")
    # Then we keep this information in the tree
    if 'np_vtx_ghost_information' in data:
      pdm_ghost_info = data['np_vtx_ghost_information']
      is_sorted = lambda a: np.all(a[:-1] <= a[1:])
      if (is_sorted(pdm_ghost_info)):
        n_vtx_unique = np.searchsorted(pdm_ghost_info,1)
        n_vtx_owned  = np.searchsorted(pdm_ghost_info,2)
        lnum_node = I.createNode(':CGNS#LocalNumbering', 'UserDefinedData_t', parent=p_zone)
        I.newDataArray('VertexSizeUnique', n_vtx_unique, parent=lnum_node)
        I.newDataArray('VertexSizeOwned', n_vtx_owned, parent=lnum_node)
    #Now create sections
    elt_section_nodes = PT.get_children_from_label(d_zone, "Elements_t")
    pdm_sections = [data[f'{j}dsections'] for j in range(4)]
    assert len(elt_section_nodes) == sum([len(sections) for sections in pdm_sections])

    # If high dim sections are first in dist tree, reverse section order
    if PT.Zone.elt_ordering_by_dim(d_zone) == -1:
      pdm_sections = pdm_sections[::-1]
    n_elt_cum = 0
    n_elt_cum_d = 0
    jumps_idx = np.cumsum([len(sections) for sections in pdm_sections])
    for i_section, section in enumerate(itertools.chain(*pdm_sections)):
      if i_section in jumps_idx: #Reset the dimension shift when changing dim
        n_elt_cum_d = 0
      elt = elt_section_nodes[i_section]
      n_i_elt = section['np_numabs'].size
      if n_i_elt > 0 or keep_empty_sections:
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
    pdm_elmt_to_cgns_elmt(part_zone, dist_zone, dims, data, options['output_connectivity'],options['keep_empty_sections'])

    output_loc = options['part_interface_loc']
    zgc_name = 'ZoneGridConnectivity#Vertex' if output_loc == 'Vertex' else 'ZoneGridConnectivity'
    zgc_created_pdm_to_cgns(part_zone, dist_zone, dims, data, output_loc, zgc_name)

    lngn_zone = MT.newGlobalNumbering(parent=part_zone)
    I.newDataArray('Vertex', data['np_vtx_ln_to_gn'], parent=lngn_zone)
    I.newDataArray('Cell', data['np_cell_ln_to_gn'], parent=lngn_zone)

    part_zones.append(part_zone)

  return part_zones
