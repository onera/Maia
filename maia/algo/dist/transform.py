import numpy as np

import maia.pytree.maia as MT

from maia.algo import transform
from maia.transfer import protocols as EP

def transform_affine_zone(zone,
                          vtx_ids,
                          comm,
                          rotation_center=np.zeros(3),
                          rotation_angle=np.zeros(3),
                          translation=np.zeros(3),
                          apply_to_fields=False):

  distri_vtx = MT.getDistribution(zone, 'Vertex')[1]
  
  all_vtx = np.arange(distri_vtx[0]+1, distri_vtx[1]+1, dtype=distri_vtx.dtype)
  PTP = EP.PartToPart([vtx_ids], [all_vtx], comm)

  vtx_mask = np.zeros(all_vtx.size, dtype=bool)
  vtx_mask[PTP.get_referenced_lnum2()[0]-1] = True

  transform.transform_affine_zone(zone, 
                                  vtx_mask,
                                  rotation_center,
                                  rotation_angle,
                                  translation,
                                  apply_to_fields)