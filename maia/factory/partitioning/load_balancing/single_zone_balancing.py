import numpy as np

def homogeneous_repart(n_elem_zone, n_part):
  """
  Split the zone in n_part homogeneous parts.
  A basic repartition, mostly usefull for debug.
  """
  step      = n_elem_zone // n_part
  remainder = n_elem_zone %  n_part

  zone_repart = step * np.ones(n_part, dtype=np.int32)  
  zone_repart[:remainder] += 1
  return zone_repart

