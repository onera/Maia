import pytest

import maia
import maia.pytree as PT
from maia.pytree.yaml          import parse_yaml_cgns

from maia.algo.dist.mesh_adaptation import unpack_metric

def test_unpack_metric():
  yz = """
  CGNSLibraryVersion CGNSLibraryVersion_t 4.2:
    Base CGNSBase_t [3, 3]:
      Zone Zone_t [[3,1,0]]:
        ZoneType ZoneType_t "Unstructured":
        FlowSol FlowSolution_t:
          Mach     DataArray_t R8 [1., 1., 1.]:
          TensorXX DataArray_t R8 [1., 1., 1.]:
          TensorZZ DataArray_t R8 [1., 1., 1.]:
          TensorXY DataArray_t R8 [1., 1., 1.]:
          TensorYY DataArray_t R8 [1., 1., 1.]:
          TensorXZ DataArray_t R8 [1., 1., 1.]:
          TensorYZ DataArray_t R8 [1., 1., 1.]:
          WrongA   DataArray_t R8 [1., 1., 1.]:
          WrongB   DataArray_t R8 [1., 1., 1.]:
          WrongC   DataArray_t R8 [1., 1., 1.]:
  """
  base = parse_yaml_cgns.to_node(yz)

  # > Wrong because leads to unexistant field
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(unpack_metric(base, "FlowSol/toto"))

  # > Wrong because leads to 3 fields
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(unpack_metric(base, "FlowSol/Wrong"))

  # > Path to unique field
  metrics_names = PT.get_names(unpack_metric(base, "FlowSol/Mach"))
  assert metrics_names==["Mach"]
  
  # > Path to multiple fields
  metrics_names = PT.get_names(unpack_metric(base, "FlowSol/Tensor"))
  assert metrics_names==[ "TensorXX","TensorXY","TensorXZ",
                          "TensorYY","TensorYZ","TensorZZ" ]

  # > Paths to multiple fields (order matters)
  metric = ["FlowSol/TensorXX", "FlowSol/TensorZZ",
            "FlowSol/TensorXZ", "FlowSol/TensorXY",
            "FlowSol/TensorYZ", "FlowSol/TensorYY"]
  metrics_names = PT.get_names(unpack_metric(base, metric))
  assert metrics_names==[ "TensorXX","TensorZZ","TensorXZ",
                          "TensorXY","TensorYZ","TensorYY" ]
