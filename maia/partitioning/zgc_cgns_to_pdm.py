import Converter.Internal as I
import numpy              as NPY

# --------------------------------------------------------------------------
def zgc_cgns_to_pdm(dist_zone):
  """
  """
  n_join = 0
  for zone_gc in I.getNodesFromType1(dist_zone, 'ZoneGridConnectivity_t'):
    gcs     = I.getNodesFromType1(zone_gc, 'GridConnectivity_t')
    n_join += len(gcs)

  joins_ids         = NPY.empty(n_join  , dtype='int32', order='F')
  dface_join_idx    = NPY.empty(n_join+1, dtype='int32', order='F')
  dface_join_idx[0] = 0
  dface_join        = list()
  for zone_gc in I.getNodesFromType1(dist_zone, 'ZoneGridConnectivity_t'):
    gcs     = I.getNodesFromType1(zone_gc, 'GridConnectivity_t')
    for i_group, join in enumerate(gcs):
      pl_n = I.getNodeFromName1(join, 'PointList')
      if pl_n is not None:
        # > Don't use I.getValue which return an int instead of np array if len(pl)=1
        pl = pl_n[1][0,:]
      else:
        pl = NPY.empty(0, dtype='int32', order='F')
      dface_join.append(pl)
      dface_join_idx[i_group+1] = dface_join_idx[i_group] + pl.shape[0]
      # joins_ids[i_group] = I.getNodeFromName1(join, 'Ordinal')[1] - 1

  if n_join > 0:
    dface_join = NPY.concatenate(dface_join)
  else:
    dface_join = None

  return dface_join, dface_join_idx, joins_ids
