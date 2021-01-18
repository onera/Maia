import Converter.Internal as I
import numpy              as NPY

from maia.utils import zone_elements_utils as EZU


def bnd_cgns_to_pdm(dist_zone):
  """
  """
  # > Find shift in NGon
  first_ngon_elmt, last_ngon_elmt = EZU.get_range_of_ngon(dist_zone)
  #print("first_ngon_elmt::", first_ngon_elmt)
  #print("last_ngon_elmt ::", last_ngon_elmt)

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
        pl = NPY.copy(pl_n[1][0,:]) - first_ngon_elmt + 1
        # Verification dans la range du ngon
      else:
        pl = NPY.empty(0, dtype='int32', order='F')
      dface_bound.append(pl)
      dface_bound_idx[i_group+1] = dface_bound_idx[i_group] + pl.shape[0]

  if n_bnd > 0:
    dface_bound = NPY.concatenate(dface_bound)
  else:
    dface_bound = None

  #print(dface_bound, dface_bound_idx)
  return dface_bound, dface_bound_idx
