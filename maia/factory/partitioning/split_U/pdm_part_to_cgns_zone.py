import numpy              as np
import itertools

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import layouts, np_utils

import Pypdm.Pypdm as PDM

def _get_part_dim(dims):
  if dims['n_cell'] > 0:
    return 3
  else:
    return 2

def dump_pdm_output(p_zone, dims, data):
  """
  Write PDM output in part_tree (for debug)
  """
  ppart_node = PT.new_child(p_zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for dim_key, dim_val in dims.items():
    PT.new_DataArray(dim_key, dim_val, parent=ppart_node)
  for data_key, data_val in data.items():
    if isinstance(data_val, np.ndarray):
      PT.new_DataArray(data_key, np.copy(data_val), parent=ppart_node)

def zgc_created_pdm_to_cgns(p_zone, d_zone, dims, data, grid_loc='FaceCenter', zgc_name='ZoneGridConnectivity'):
  """
  Create by splitting
  """
  if grid_loc not in ['FaceCenter', 'Vertex']:
    raise NotImplementedError("Unvalid specified entity")
  if grid_loc == 'FaceCenter':
    entity = 'face' if _get_part_dim(dims) == 3 else 'edge'
    _grid_loc = f"{entity.capitalize()}Center"
  else:
    entity = 'vtx'
    _grid_loc = 'Vertex'

  entity_part_bound_proc_idx = data['np_{0}_part_bound_proc_idx'.format(entity)]
  entity_part_bound_part_idx = data['np_{0}_part_bound_part_idx'.format(entity)]
  entity_part_bound_tmp      = data['np_{0}_part_bound'         .format(entity)]

  entity_part_bound = entity_part_bound_tmp.reshape((4, entity_part_bound_tmp.shape[0]//4), order='F')
  entity_part_bound = entity_part_bound.transpose()

  zgc_n = PT.new_node(zgc_name, 'ZoneGridConnectivity_t', parent=p_zone)

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

      cur_rank, cur_part = MT.conv.get_part_suffix(PT.get_name(p_zone))
      gcname = MT.conv.name_intra_gc(cur_rank, cur_part, opp_rank, opp_part)
      join_n = PT.new_GridConnectivity(name       = gcname,
                                       donor_name = MT.conv.add_part_suffix(PT.get_name(d_zone), opp_rank, opp_part),
                                       type       = 'Abutting1to1',
                                       loc        = _grid_loc,
                                       parent     = zgc_n)

      PT.new_PointList(name='PointList'     , value=pl , parent=join_n)
      PT.new_PointList(name='PointListDonor', value=pld, parent=join_n)


def pdm_vtx_to_cgns_grid_coordinates(p_zone, dims, data):
  """
  """
  coords = data['np_vtx_coord']
  fields = {'CoordinateX' : coords[0::3], 'CoordinateY' : coords[1::3], 'CoordinateZ' : coords[2::3]}
  grid_c = PT.new_GridCoordinates(fields=fields, parent=p_zone)

def pdm_renumbering_data(p_zone, data):
  color_data = PT.new_node('maia#Renumbering', 'UserDefinedData_t')
  for entity in ['cell', 'face', 'edge', 'vtx', 'thread', 'hyperplane']:
    array = data[f'np_{entity}_color']
    if array is not None:
      idx, val = np_utils.compress(array)
      PT.new_DataArray(f"{entity.capitalize()}ColorIdx", idx, parent=color_data)
      PT.new_DataArray(f"{entity.capitalize()}Color", val, parent=color_data)
  if len(PT.get_children(color_data)) > 0:
    PT.add_child(p_zone, color_data)

def pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data, connectivity_as="Element", keep_empty_sections=False):
  """
  """
  ngon_zone = [e for e in PT.iter_children_from_label(d_zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n'] != []
  if  ngon_zone or connectivity_as == 'NGon':
    # Use default name if none in tree
    nedge_name = 'EdgeElements'
    ngon_name  = 'NGonElements'
    nface_name = 'NFaceElements'
    for elt in PT.iter_children_from_label(d_zone, 'Elements_t'):
      if PT.Element.CGNSName(elt) == 'BAR_2':
        nedge_name = PT.get_name(elt)
      if PT.Element.CGNSName(elt) == 'NGON_n':
        ngon_name = PT.get_name(elt)
      elif PT.Element.CGNSName(elt) == 'NFACE_n':
        nface_name = PT.get_name(elt)

    n_face = dims['n_face']
    n_cell = dims['n_cell']
    if _get_part_dim(dims) == 3:
      ngon_er  = np.array([1, n_face], np.int32)
      if ngon_zone:
        ngon_eso = data['np_face_vtx_idx']
        ngon_ec  = data['np_face_vtx']
      else: #When coming from elements, we have no face_vtx; rebuild it
        ngon_eso = data['np_face_edge_idx']
        ngon_ec  = PDM.compute_face_vtx_from_face_and_edge(data['np_face_edge_idx'],
                                                           data['np_face_edge'],
                                                           data['np_edge_vtx'])

      ngon_pe = np.empty((n_face, 2), dtype=np.int32, order='F')
      layouts.pdm_face_cell_to_pe_cgns(data['np_face_cell'], ngon_pe)
      np_utils.shift_nonzeros(ngon_pe, n_face)

      nface_er  = np.array([1, n_cell], np.int32) + n_face
      nface_eso = data['np_cell_face_idx']
      nface_ec  = data['np_cell_face']

      ngon_n = PT.new_NGonElements(ngon_name, parent=p_zone, erange=ngon_er, eso=ngon_eso, ec=ngon_ec, pe=ngon_pe)
      nface_n = PT.new_NFaceElements(nface_name, parent=p_zone, erange=nface_er, eso=nface_eso, ec=nface_ec)
      MT.newGlobalNumbering({'Element' : data['np_face_ln_to_gn']}, ngon_n)
      MT.newGlobalNumbering({'Element' : data['np_cell_ln_to_gn']}, nface_n)

    elif _get_part_dim(dims) == 2:
      face_edge_idx = data['np_face_edge_idx']   
      face_edge     = data['np_face_edge']   
      edge_vtx      = data['np_edge_vtx']   

      n_edge = edge_vtx.size//2 
      edge_vtx_idx  = 2*np.arange(n_edge+1, dtype=np.int32)
      edge_face_idx, edge_face = PDM.connectivity_transpose(n_edge, face_edge_idx, face_edge)
      assert edge_face_idx.size - 1 == n_edge
      edge_er = np.array([1, n_edge], np.int32)
      nedge_pe = np.empty((n_edge,2), np.int32, order='F')
      layouts.strided_connectivity_to_pe(edge_face_idx, edge_face, nedge_pe)
      np_utils.shift_nonzeros(nedge_pe, n_edge)

      ngon_er = np.array([1, n_face], np.int32) + n_edge
      ngon_eso = face_edge_idx
      ngon_ec  = PDM.compute_face_vtx_from_face_and_edge(face_edge_idx, face_edge, edge_vtx)

      nedge_n = PT.new_Elements(nedge_name, 'BAR_2', erange=edge_er, econn=edge_vtx, parent=p_zone)
      ngon_n = PT.new_NGonElements(ngon_name, parent=p_zone, erange=ngon_er, eso=ngon_eso, ec=ngon_ec)
      PT.new_DataArray('ParentElements', nedge_pe, parent=nedge_n)
      MT.newGlobalNumbering({'Element' : data['np_edge_ln_to_gn']}, nedge_n)
      MT.newGlobalNumbering({'Element' : data['np_face_ln_to_gn']}, ngon_n)

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
        lnum_node = PT.new_node(':CGNS#LocalNumbering', 'UserDefinedData_t', parent=p_zone)
        PT.new_DataArray('VertexSizeUnique', n_vtx_unique, parent=lnum_node)
        PT.new_DataArray('VertexSizeOwned', n_vtx_owned, parent=lnum_node)
    #Now create sections
    elt_section_nodes = PT.Zone.get_ordered_elements(d_zone)
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
        elt_n = PT.new_Elements(PT.get_name(elt), PT.get_value(elt), parent=p_zone,
            erange = [n_elt_cum+1, n_elt_cum+n_i_elt], econn = section['np_connec'])
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
        lnum_node = PT.new_node(':CGNS#LocalNumbering', 'UserDefinedData_t', parent=elt_n)
        PT.new_DataArray('ExplicitEntity', section['np_parent_num'], parent=lnum_node)

        MT.newGlobalNumbering(numberings, elt_n)

      n_elt_cum   += n_i_elt
      n_elt_cum_d += PT.Element.Size(elt_section_nodes[i_section])

def pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm, options):
  """
  """
  #Dims and data should be related to the dist zone and of size n_parts
  part_zones = list()
  for i_part, (dims, data) in enumerate(zip(l_dims, l_data)):

    base_dim = 2 if (dims['n_cell'] == 0 and dims['n_face'] > 0) else 3
    cell_key = {3: 'n_cell', 2: 'n_face'}[base_dim]
    part_zone = PT.new_Zone(name  = MT.conv.add_part_suffix(PT.get_name(dist_zone), comm.Get_rank(), i_part),
                           size = [[dims['n_vtx'],dims[cell_key],0]],
                           type = 'Unstructured')

    if options['dump_pdm_output']:
      dump_pdm_output(part_zone, dims, data)
    pdm_vtx_to_cgns_grid_coordinates(part_zone, dims, data)
    pdm_elmt_to_cgns_elmt(part_zone, dist_zone, dims, data, options['output_connectivity'],options['keep_empty_sections'])

    output_loc = options['part_interface_loc']
    zgc_name = 'ZoneGridConnectivity'
    zgc_created_pdm_to_cgns(part_zone, dist_zone, dims, data, output_loc, zgc_name)

    pdm_renumbering_data(part_zone, data)

    cell_lngn_key = {3: 'np_cell_ln_to_gn', 2: 'np_face_ln_to_gn'}[base_dim]
    lngn_zone = MT.newGlobalNumbering(parent=part_zone)
    PT.new_DataArray('Vertex', data['np_vtx_ln_to_gn'], parent=lngn_zone)
    PT.new_DataArray('Cell', data[cell_lngn_key], parent=lngn_zone)

    part_zones.append(part_zone)

  return part_zones
