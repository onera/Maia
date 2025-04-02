import pytest
import pytest_parallel

import maia
import maia.pytree as PT

from maia.algo.dist import fsdm_distribution as FSD

@pytest_parallel.mark.parallel(2)
def test_add_fsdm_distribution(comm):
  # PartTree generated from TETRA_4 (n=14) with SORT_INT_EXT
  if comm.rank == 0:
    ptree = PT.yaml.to_cgns_tree("""
    zone.P0.N0 Zone_t [[1463, 5443, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      :CGNS#LocalNumbering UserDefinedData_t:
        VertexSizeUnique DataArray_t [1265]:
        VertexSizeOwned DataArray_t [1364]:
      TETRA_4.0 Elements_t [10, 0]:
        ElementRange IndexRange_t [1, 5443]:
      TRI_3.0 Elements_t [5, 0]:
        ElementRange IndexRange_t [5444, 6473]:                 
    """)
  elif comm.rank == 1:
    ptree = PT.yaml.to_cgns_tree("""
    zone.P1.N0 Zone_t [[1479, 5542, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      :CGNS#LocalNumbering UserDefinedData_t:
        VertexSizeUnique DataArray_t [1281]:
        VertexSizeOwned DataArray_t [1380]:
      TETRA_4.0 Elements_t [10, 0]:
        ElementRange IndexRange_t [1, 5542]:
      TRI_3.0 Elements_t [5, 0]:
        ElementRange IndexRange_t [5543, 6540]:
    """)

  FSD.add_fsdm_distribution(ptree, comm)

  pzone = PT.get_node_from_label(ptree, 'Zone_t')
  if comm.rank == 0:
    assert (PT.get_node_from_path(pzone, ':CGNS#Distribution/Vertex')[1] == [0, 1364, 2744]).all()
    assert (PT.get_node_from_path(pzone, 'TETRA_4.0/:CGNS#Distribution/Element')[1] == [0, 5443, 10985]).all()
    assert (PT.get_node_from_path(pzone, 'TRI_3.0/:CGNS#Distribution/Element')[1] == [0, 1030, 2028]).all()
  elif comm.rank == 1:
    assert (PT.get_node_from_path(pzone, ':CGNS#Distribution/Vertex')[1] == [1364, 2744, 2744]).all()
    assert (PT.get_node_from_path(pzone, 'TETRA_4.0/:CGNS#Distribution/Element')[1] == [5443, 10985, 10985]).all()
    assert (PT.get_node_from_path(pzone, 'TRI_3.0/:CGNS#Distribution/Element')[1] == [1030, 2028, 2028]).all()
  

@pytest_parallel.mark.parallel(1)
def test_multidom_fail(comm):
  ptree = PT.yaml.to_cgns_tree("""
  Dom1.P0.N0 Zone_t:
  Dom2.P0.N0 Zone_t:
  """)
  with pytest.raises(RuntimeError, match='only one zone per process'):
    FSD.add_fsdm_distribution(ptree, comm)

