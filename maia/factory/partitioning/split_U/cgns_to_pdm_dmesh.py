import numpy          as np
from Pypdm.Pypdm import DistributedMesh

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, layouts, as_pdm_gnum
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
#from maia.transfer.dist_to_part.index_exchange import collect_distributed_pl

def cgns_dist_zone_to_pdm_dmesh(dist_zone, comm):
  """
  Create a pdm_dmesh structure from a distributed zone
  """
  distrib_vtx      = PT.get_value(MT.getDistribution(dist_zone, 'Vertex'))
  distrib_cell     = PT.get_value(MT.getDistribution(dist_zone, 'Cell'))

  # > Try to hook NGon
  ngon_node = PT.Zone.NGonNode(dist_zone)
  ngon_first = PT.Element.Range(ngon_node)[0] == 1
  dface_vtx = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1])
  ngon_pe   = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ParentElements'     )[1])
  ngon_eso  = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ElementStartOffset' )[1])

  distrib_face     = as_pdm_gnum(PT.get_value(MT.getDistribution(ngon_node, 'Element')))
  distrib_face_vtx = as_pdm_gnum(PT.get_value(MT.getDistribution(ngon_node, 'ElementConnectivity')))

  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  dn_cell = distrib_cell[1] - distrib_cell[0]
  dn_face = distrib_face[1] - distrib_face[0]
  dn_edge = -1 #Not used

  if dn_vtx > 0:
    gridc_n    = PT.get_child_from_name(dist_zone, 'GridCoordinates')
    cx         = PT.get_child_from_name(gridc_n, 'CoordinateX')[1]
    cy         = PT.get_child_from_name(gridc_n, 'CoordinateY')[1]
    cz         = PT.get_child_from_name(gridc_n, 'CoordinateZ')[1]
    dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
  else:
    dvtx_coord = np.empty(0, dtype='float64', order='F')

  if dn_face > 0:
    dface_cell    = np.empty(2*dn_face  , dtype=pdm_gnum_dtype) # Respect pdm_gnum_type
    layouts.pe_cgns_to_pdm_face_cell(ngon_pe      , dface_cell      )
    # PDM expects a PE in local cell indexing, shift is needed
    if ngon_first:
      np_utils.shift_nonzeros(dface_cell, -distrib_face[2])
    dface_vtx_idx = np.add(ngon_eso, -distrib_face_vtx[0], dtype=np.int32) #Local index is int32bits
    
  else:
    dface_vtx_idx = np.zeros(1, dtype=np.int32    )
    dface_vtx     = np.empty(0, dtype=pdm_gnum_dtype)
    dface_cell    = np.empty(0, dtype=pdm_gnum_dtype)

  # > Prepare bnd
  #bc_point_lists = collect_distributed_pl(dist_zone, ['ZoneBC_t/BC_t'])
  #dface_bound_idx, dface_bound = np_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)
  dface_bound_idx = np.zeros(1, dtype=np.int32)
  dface_bound     = np.empty(0, dtype=pdm_gnum_dtype)
  # > Find shift in NGon
  # first_ngon_elmt, last_ngon_elmt = PT.Zone.get_range_of_ngon(dist_zone)
  # dface_bound = dface_bound - first_ngon_elmt + 1

  # > Prepare joins
  # gc_type_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
  # gc_point_lists = collect_distributed_pl(dist_zone, [gc_type_path])
  # dface_join_idx, dface_join = np_utils.concatenate_point_list(gc_point_lists, pdm_gnum_dtype)
  # joins_ids = [PT.get_child_from_name(gc, 'Ordinal')[1][0] for gc in \
      # PT.iter_children_from_predicates(dist_zone, gc_type_path)]
  # joins_ids = np.array(joins_ids, dtype='int32') - 1
  joins_ids      = np.empty(0, dtype=np.int32)
  dface_join_idx = np.zeros(1, dtype=np.int32)
  dface_join     = np.empty(0, dtype=pdm_gnum_dtype)

  n_bnd  = dface_bound_idx.shape[0] - 1
  n_join = dface_join_idx.shape[0]  - 1

  dmesh = DistributedMesh(comm, dn_cell, dn_face, dn_edge, dn_vtx, n_bnd, n_join)
  dmesh.dmesh_set(dvtx_coord, dface_vtx_idx, dface_vtx, dface_cell,
                  dface_bound_idx, dface_bound, joins_ids,
                  dface_join_idx, dface_join)

  # > Create an older --> To Suppress after all
  multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  PT.new_DataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)
  PT.new_DataArray('dface_vtx_idx'  , dface_vtx_idx  , parent=multi_part_node)
  PT.new_DataArray('dface_vtx'      , dface_vtx      , parent=multi_part_node)
  PT.new_DataArray('dface_cell'     , dface_cell     , parent=multi_part_node)
  PT.new_DataArray('dface_bound_idx', dface_bound_idx, parent=multi_part_node)
  PT.new_DataArray('dface_bound'    , dface_bound    , parent=multi_part_node)
  PT.new_DataArray('joins_ids'      , joins_ids      , parent=multi_part_node)
  PT.new_DataArray('dface_join_idx' , dface_join_idx , parent=multi_part_node)
  PT.new_DataArray('dface_join'     , dface_join     , parent=multi_part_node)

  return dmesh

