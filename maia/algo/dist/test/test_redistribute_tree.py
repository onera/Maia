import numpy as np
import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.utils         import par_utils
from   maia.algo.dist     import redistribute_tree as RDT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'

# =======================================================================================
# ---------------------------------------------------------------------------------------

@mark_mpi_test([1,3])
def test_redistribute_pl_node_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t {dtype} [1, 5, 7, 12]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [1, 2, 3, 4]:
    """
  if sub_comm.Get_rank() == 1:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t {dtype} [8, 11, 10, 21]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [5, 6, 7, 8]:
    """
  if sub_comm.Get_rank() == 2:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t {dtype} [22, 23, 30]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [9, 10, 11]:
    """

  dist_bc = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]

  if sub_comm.Get_rank()==0:
    import Converter.Internal as I
    I.printTree(dist_bc)

  gather_bc = RDT.redistribute_pl_node(dist_bc, par_utils.gathering_distribution, sub_comm)

  if sub_comm.Get_rank()==0:
    assert PT.get_node_from_name(gather_bc, 'Index'    ) == np.arange(1, 12)
    assert PT.get_node_from_name(gather_bc, 'PointList') == np.array([1, 5, 7, 12, 8, 11, 10, 21, 22, 23, 30])

  else:
    assert PT.get_node_from_name(gather_bc, 'Index'    ) == np.empty(0, dtype=np.int32)
    assert PT.get_node_from_name(gather_bc, 'PointList') == np.empty(0, dtype=np.int32)

# ---------------------------------------------------------------------------------------
# =======================================================================================