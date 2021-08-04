import pytest
import numpy as np

from maia.utils import parse_yaml_cgns
from maia.transform import duplicate

# import Converter.Internal as I


###############################################################################
class Test_duplicate_zone_with_transformation():
  yz = """
       Zone Zone_t I4 [[18,4,0]]:
         ZoneType ZoneType_t "Unstructured":
         GridCoordinates GridCoordinates_t:
           CoordinateX DataArray_t:
             R8 : [ 0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2 ]
           CoordinateY DataArray_t:
             R4 : [ 0,0,0,
                    1,1,1,
                    2,2,2,
                    0,0,0,
                    1,1,1,
                    2,2,2 ]
           CoordinateZ DataArray_t:
             R4 : [ 0,0,0,
                    0,0,0,
                    0,0,0,
                    1,1,1,
                    1,1,1,
                    1,1,1 ]
       """
  zone = parse_yaml_cgns.to_node(yz)
  # --------------------------------------------------------------------------- #
  def test_duplicate_zone_with_transformation_simple_none(self):
    duplicated_zone = duplicate.duplicate_zone_with_transformation(self.zone,"DuplicatedZone")
    assert(duplicated_zone[0]=="DuplicatedZone")
    expected_yz = """
                  DuplicatedZone Zone_t I4 [[18,4,0]]:
                    ZoneType ZoneType_t "Unstructured":
                    GridCoordinates GridCoordinates_t:
                      CoordinateX DataArray_t:
                        R8 : [ 0,1,2,
                               0,1,2,
                               0,1,2,
                               0,1,2,
                               0,1,2,
                               0,1,2 ]
                      CoordinateY DataArray_t:
                        R4 : [ 0,0,0,
                               1,1,1,
                               2,2,2,
                               0,0,0,
                               1,1,1,
                               2,2,2 ]
                      CoordinateZ DataArray_t:
                        R4 : [ 0,0,0,
                               0,0,0,
                               0,0,0,
                               1,1,1,
                               1,1,1,
                               1,1,1 ]
                  """
    # assert(cgns_to_yaml(duplicated_zone)==expected_yz) #TO DO : conversion pytree_to_yaml
