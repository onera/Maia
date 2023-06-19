import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import maia
import maia.pytree as PT
from maia.algo.mesh_adaptation import unpack_metric
from maia.pytree.yaml          import parse_yaml_cgns

# @mark_mpi_test(1)
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
  metric = "Base/Zone/FlowSol/toto"
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(unpack_metric(base, metric))

  # > Wrong because leads to 3 fields
  metric = "Base/Zone/FlowSol/Wrong"
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(unpack_metric(base, metric))

  # > Path to unique field
  metric = "Base/Zone/FlowSol/Mach"
  metrics_names = PT.get_names(unpack_metric(base, metric))
  assert metrics_names==["Mach"]
  
  # > Path to multiple fields
  metric = "Base/Zone/FlowSol/Tensor"
  metrics_names = PT.get_names(unpack_metric(base, metric))
  assert metrics_names==[ "TensorXX","TensorXY","TensorXZ",
                          "TensorYY","TensorYZ","TensorZZ" ]

  # > Paths to multiple fields (order matters)
  metric = ["Base/Zone/FlowSol/TensorXX", "Base/Zone/FlowSol/TensorZZ",
            "Base/Zone/FlowSol/TensorXZ", "Base/Zone/FlowSol/TensorXY",
            "Base/Zone/FlowSol/TensorYZ", "Base/Zone/FlowSol/TensorYY"]
  metrics_names = PT.get_names(unpack_metric(base, metric))
  assert metrics_names==[ "TensorXX","TensorZZ","TensorXZ",
                          "TensorXY","TensorYZ","TensorYY" ]