import Converter.Internal as I
import maia.sids.sids as SIDS
import numpy          as NPY
from maia.connectivity import connectivity_transform as CNT
from .bnd_cgns_to_pdm import bnd_cgns_to_pdm
from .zgc_cgns_to_pdm import zgc_cgns_to_pdm

import Pypdm.Pypdm as PDM

# --------------------------------------------------------------------------
def cgns_dist_zone_to_pdm_dmesh(dist_zone):
  """
  """
  distrib_ud       = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
  distrib_vtx      = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell     = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  # > Try to hooks NGon
  ngon_n = None
  for elmt in I.getNodesFromType1(dist_zone, 'Elements_t'):
    if(elmt[1][0] == 22):
      found    = True
      dface_vtx = I.getNodeFromName1(elmt, 'ElementConnectivity')[1]
      ngon_pe   = I.getNodeFromName1(elmt, 'ParentElements'     )[1]
      ngon_eso  = I.getNodeFromName1(elmt, 'ElementStartOffset' )[1]

      distrib_ngon_ud  = I.getNodeFromName1(elmt           , ':CGNS#Distribution')
      distrib_face     = I.getNodeFromName1(distrib_ngon_ud, 'Distribution'      )[1]
      distrib_face_vtx = I.getNodeFromName1(distrib_ngon_ud, 'DistributionElementConnectivity')[1]
  if(not found):
    raise NotImplementedError

  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  dn_cell = distrib_cell[1] - distrib_cell[0]
  dn_face = distrib_face[1] - distrib_face[0]

  if dn_vtx > 0:
    gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
    cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
    cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
    dvtx_coord = NPY.hstack(list(zip(cx, cy, cz)))
  else:
    dvtx_coord = NPY.empty(0, dtype='float64', order='F')

  if dn_face > 0:
    dface_cell    = NPY.empty( 2*dn_face  , dtype=ngon_pe.dtype )
    dface_vtx_idx = NPY.empty(   dn_face+1, dtype=NPY.int32     ) # Local index is int32bits
    CNT.pe_cgns_to_pdm_face_cell(ngon_pe      , dface_cell      )
    CNT.compute_idx_local       (dface_vtx_idx, ngon_eso, distrib_face_vtx)
  else:
    dface_vtx_idx = NPY.zeros(1, dtype=ngon_pe.dtype, order='F')
    dface_vtx     = NPY.empty(0, dtype=ngon_pe.dtype, order='F')
    dface_cell    = NPY.empty(0, dtype=ngon_pe.dtype, order='F')
  # LOG.debug(" dface_vtx    = {0}".format(dface_vtx   ))
  # LOG.debug(" dface_vtx_idx = {0}".format(dface_vtx_idx))
  # LOG.debug(" dface_cell   = {0}".format(dface_cell  ))

  # > Prepare  bnd
  dface_bound, dface_bound_idx = bnd_cgns_to_pdm(dist_zone)
  n_bnd = dface_bound_idx.shape[0]-1
  # LOG.debug("nBoundary     = {0}".format(n_bnd))
  # LOG.debug("dface_bound    = {0}".format(dface_bound))
  # LOG.debug("dface_bound_idx = {0}".format(dface_bound_idx))

  # > Prepare joins
  dface_join, dface_join_idx, joins_ids = zgc_cgns_to_pdm(dist_zone)
  n_join = dface_join_idx.shape[0]-1
  # LOG.debug("n_join         = {0}".format(n_join))
  # LOG.debug("dface_join     = {0}".format(dface_join))
  # LOG.debug("dface_join_idx = {0}".format(dface_join_idx))
  # LOG.debug("joins_ids      = {0}".format(joins_ids))

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  dmesh = PDM.DistributedMesh(dn_cell, dn_face, dn_vtx, n_bnd, n_join)
  dmesh.dmesh_set(dvtx_coord, dface_vtx_idx, dface_vtx, dface_cell,
                  dface_bound_idx, dface_bound, joins_ids,
                  dface_join_idx, dface_join)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # print 'Avant sortie : ', dface_vtx
  # del(dface_vtx)
  # del(dface_vtx_idx)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return dmesh

# --------------------------------------------------------------------------
def cgns_dist_tree_to_joinopp_array(dist_tree):
  """
  """
  zones = I.getZones(dist_tree)

  jns = []
  for zone in zones:
    # > Get ZoneGridConnectivity List
    zone_gcs = I.getNodesFromType1(zone, 'ZoneGridConnectivity_t')
    # > Get Join List if ZoneGridConnectivity is not None
    #   - Match Structured and Match Hybride
    if (zone_gcs != []):
      jns += I.getNodesFromType1(zone_gcs, 'GridConnectivity_t')
      jns += I.getNodesFromType1(zone_gcs, 'GridConnectivity1to1_t')
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Count joins and declare array
  join_to_opp = NPY.empty(len(jns), dtype='int32' )
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Fill array
  for jn in jns:
    join_id     = I.getNodeFromName1(jn, 'Ordinal')[1]
    join_opp_id = I.getNodeFromName1(jn, 'OrdinalJoinOpp')[1]
    join_to_opp[join_id - 1] = join_opp_id - 1
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return join_to_opp
