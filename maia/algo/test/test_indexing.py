import pytest
import pytest_parallel
import numpy as np

from maia.pytree.yaml   import parse_yaml_cgns

from maia.algo import indexing

def test_get_ngon_pe_local():
  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[9, 0], [10, 0], [11, 12], [0, 12]]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  assert (indexing.get_ngon_pe_local(ngon) == np.array([[9-8,0], [10-8,0], [11-8,12-8], [0,12-8]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[1, 0], [2, 0], [3, 4], [0, 4]]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  assert (indexing.get_ngon_pe_local(ngon) == np.array([[1,0], [2,0], [3,4], [0,4]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  with pytest.raises(RuntimeError):
    indexing.get_ngon_pe_local(ngon)

