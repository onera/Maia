import Converter.Internal as I
import maia.sids.sids as SIDS
import maia.sids.Internal_ext as IE
import numpy          as np
from maia.connectivity import connectivity_transform as CNT
from maia.sids  import elements_utils as EU
from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
#from maia.tree_exchange.dist_to_part.index_exchange import collect_distributed_pl
from Pypdm.Pypdm import DistributedMesh

def cgns_dist_zone_to_pdm_dmesh(dist_zone, comm):
  """
  Create a pdm_dmesh structure from a distributed zone
  """
  distrib_vtx      = I.getVal(IE.getDistribution(dist_zone, 'Vertex'))
  distrib_cell     = I.getVal(IE.getDistribution(dist_zone, 'Cell'))

  # > Try to hook NGon
  found = False
  for elt in I.getNodesFromType1(dist_zone, 'Elements_t'):
    if SIDS.ElementType(elt) == 22:
      found    = True
      dface_vtx = I.getNodeFromName1(elt, 'ElementConnectivity')[1].astype(pdm_gnum_dtype)
      ngon_pe   = I.getNodeFromName1(elt, 'ParentElements'     )[1].astype(pdm_gnum_dtype)
      ngon_eso  = I.getNodeFromName1(elt, 'ElementStartOffset' )[1].astype(pdm_gnum_dtype)

      distrib_face     = I.getVal(IE.getDistribution(elt, 'Element')).astype(pdm_gnum_dtype)
      distrib_face_vtx = I.getVal(IE.getDistribution(elt, 'ElementConnectivity')).astype(pdm_gnum_dtype)
  if not found :
    raise RuntimeError

  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  dn_cell = distrib_cell[1] - distrib_cell[0]
  dn_face = distrib_face[1] - distrib_face[0]
  dn_edge = -1 #Not used

  if dn_vtx > 0:
    gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
    cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
    cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
    dvtx_coord = py_utils.interweave_arrays([cx,cy,cz])
  else:
    dvtx_coord = np.empty(0, dtype='float64', order='F')

  if dn_face > 0:
    dface_cell    = np.empty(2*dn_face  , dtype=pdm_gnum_dtype) # Respect pdm_gnum_type
    dface_vtx_idx = np.empty(  dn_face+1, dtype=np.int32     ) # Local index is int32bits
    CNT.pe_cgns_to_pdm_face_cell(ngon_pe      , dface_cell      )
    CNT.compute_idx_local       (dface_vtx_idx, ngon_eso, distrib_face_vtx)
  else:
    dface_vtx_idx = np.zeros(1, dtype=np.int32    )
    dface_vtx     = np.empty(0, dtype=pdm_gnum_dtype)
    dface_cell    = np.empty(0, dtype=pdm_gnum_dtype)

  # > Prepare bnd
  #bc_point_lists = collect_distributed_pl(dist_zone, ['ZoneBC_t/BC_t'])
  #dface_bound_idx, dface_bound = py_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)
  dface_bound_idx = np.zeros(1, dtype=np.int32)
  dface_bound     = np.empty(0, dtype=pdm_gnum_dtype)
  # > Find shift in NGon
  # first_ngon_elmt, last_ngon_elmt = EU.get_range_of_ngon(dist_zone)
  # dface_bound = dface_bound - first_ngon_elmt + 1

  # > Prepare joins
  # gc_type_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
  # gc_point_lists = collect_distributed_pl(dist_zone, [gc_type_path])
  # dface_join_idx, dface_join = py_utils.concatenate_point_list(gc_point_lists, pdm_gnum_dtype)
  # joins_ids = [I.getNodeFromName1(gc, 'Ordinal')[1][0] for gc in \
      # IE.iterNodesByMatching(dist_zone, gc_type_path)]
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
  multi_part_node = I.createUniqueChild(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  I.newDataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)
  I.newDataArray('dface_vtx_idx'  , dface_vtx_idx  , parent=multi_part_node)
  I.newDataArray('dface_vtx'      , dface_vtx      , parent=multi_part_node)
  I.newDataArray('dface_cell'     , dface_cell     , parent=multi_part_node)
  I.newDataArray('dface_bound_idx', dface_bound_idx, parent=multi_part_node)
  I.newDataArray('dface_bound'    , dface_bound    , parent=multi_part_node)
  I.newDataArray('joins_ids'      , joins_ids      , parent=multi_part_node)
  I.newDataArray('dface_join_idx' , dface_join_idx , parent=multi_part_node)
  I.newDataArray('dface_join'     , dface_join     , parent=multi_part_node)

  return dmesh

