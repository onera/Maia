import Converter.Internal as I
import numpy              as NPY


def bnd_cgns_to_pdm(dist_zone):
  """
  """
  n_bnd = 0
  for zone_bc in I.getNodesFromType1(dist_zone, 'ZoneBC_t'):
    bcs = I.getNodesFromType1(zone_bc, 'BC_t')
    n_bnd += len(bcs)

  dface_bound        = list()
  dface_bound_idx    = NPY.empty(n_bnd+1, dtype='int32', order='F')
  dface_bound_idx[0] = 0
  for zone_bc in I.getNodesFromType1(dist_zone, 'ZoneBC_t'):
    bcs = I.getNodesFromType1(zone_bc, 'BC_t')
    for i_group, bc in enumerate(bcs):
      pl_n = I.getNodeFromName1(bc, 'PointList')
      if pl_n is not None:
        # > Don't use I.getValue which return an int instead of np array if len(PL)=1
        pl = pl_n[1][0,:]
      else:
        pl = NPY.empty(0, dtype='int32', order='F')
      dface_bound.append(pl)
      dface_bound_idx[i_group+1] = dface_bound_idx[i_group] + pl.shape[0]

  if n_bnd > 0:
    dface_bound = NPY.concatenate(dface_bound)
  else:
    dface_bound = None

  return dface_bound, dface_bound_idx
