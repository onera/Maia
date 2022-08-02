import numpy as np

import Converter.Internal       as I
import maia.pytree as PT
from maia.utils import py_utils, np_utils

def transform_zone(zone,
                   rotation_center = np.zeros(3),
                   rotation_angle  = np.zeros(3),
                   translation     = np.zeros(3),
                   apply_to_fields = False):
  """
  Apply the affine transformation to the coordinates of the given zone.
  If apply_to_fields is True, also rotate all the vector fields in CGNS nodes of type
  "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"
  """
  # Transform coords
  for grid_co in PT.iter_children_from_label(zone, "GridCoordinates_t"):
    coords_n = [I.getNodeFromName1(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z']]
    coords = [I.getVal(n) for n in coords_n]
  
    tr_coords = np_utils.transform_cart_vectors(*coords, translation, rotation_center, rotation_angle)
    for coord_n, tr_coord in zip(coords_n, tr_coords):
      I.setValue(coord_n, tr_coord)

  # Transform fields
  if apply_to_fields:
    fields_nodes  = PT.get_children_from_label(zone, "FlowSolution_t")
    fields_nodes += PT.get_children_from_label(zone, "DiscreteData_t")
    fields_nodes += PT.get_children_from_label(zone, "ZoneSubRegion_t")
    for bc in PT.iter_children_from_predicates(zone, "ZoneBC_t/BC_t"):
      fields_nodes += PT.get_children_from_label(bc, "BCDataSet_t")
    for fields_node in fields_nodes:
      data_names = [I.getName(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
      cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names)
      for basename in cartesian_vectors_basenames:
        vectors_n = [I.getNodeFromNameAndType(fields_node, f"{basename}{c}", 'DataArray_t')  for c in ['X', 'Y', 'Z']]
        vectors = [I.getVal(n) for n in vectors_n]
        # Assume that vectors are position independant
        # Be careful, if coordinates vector needs to be transform, the translation is not applied !
        tr_vectors = np_utils.transform_cart_vectors(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
        for vector_n, tr_vector in zip(vectors_n, tr_vectors):
          I.setValue(vector_n, tr_vector)
