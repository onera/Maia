import pytest_parallel

import numpy as np
from mpi4py import MPI

import Pypdm.Pypdm as PDM
import maia.pytree as PT

from maia                      import npy_pdm_gnum_dtype as pdm_dtype
from maia.pytree.yaml          import parse_yaml_cgns
from maia.factory.partitioning.split_U import part_all_zones as partU

def test_prepare_part_weight():
  zones = [PT.new_Zone('ZoneA', type='Unstructured'),
           PT.new_Zone('ZoneB', type='Unstructured'),
           PT.new_Zone('ZoneC', type='Unstructured')]

  base_to_blocks  = {'Base' : zones}
  d_zone_to_parts = {'Base/ZoneA' : [.3], 'Base/ZoneB' : [], 'Base/ZoneC' : [.2,.5,.3]}
  n_part_per_zone, part_weight = partU.prepare_part_weight(base_to_blocks, d_zone_to_parts)
  assert (n_part_per_zone == [1,0,3]).all()
  assert (part_weight == [.3,.2,.5,.3]).all()

  base_to_blocks  = {'Base' : [zones[0]], 'Base2' : zones[1:]}
  d_zone_to_parts = {'Base2/ZoneC' : [.2,.5,.3], 'Base/ZoneA' : [.3], 'Base2/ZoneB' : []}
  n_part_per_zone, part_weight = partU.prepare_part_weight(base_to_blocks, d_zone_to_parts)
  assert (n_part_per_zone == [1,0,3]).all()
  assert (part_weight == [.3,.2,.5,.3]).all()

def test_set_mpart_reordering():
  keep_alive = []
  reorder_options = {'cell_renum_method' : 'CUTHILL', 'face_renum_method' : 'LEXICOGRAPHIC', 'vtx_renum_method' : 'SORT_INT_EXT'}
  n_part_per_zones = np.array([1,2], dtype=np.int32)
  mpart = PDM.MultiPart(2, np.array([1,2], dtype=np.int32), 0, 1, 1, None, MPI.COMM_SELF)
  partU.set_mpart_reordering(mpart, reorder_options, keep_alive)

@pytest_parallel.mark.parallel(2)
def test_set_mpart_dmeshes(comm):
  dtype = 'I4' if pdm_dtype == np.int32 else 'I8'
  dt = f"""
ZoneA Zone_t [[1,1,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 []:
    CoordinateY DataArray_t R8 []:
    CoordinateZ DataArray_t R8 []:
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t [1, 1]:
    ElementConnectivity DataArray_t {dtype} []:
    ElementStartOffset DataArray_t {dtype} [0]:
    ParentElements DataArray_t {dtype} [[],[]]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [0,0,0]:
      ElementConnectivity DataArray_t {dtype} [0,0,0]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0,0,0]:
    Cell DataArray_t {dtype} [0,0,0]:
ZoneB Zone_t [[1,1,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 []:
    CoordinateY DataArray_t R8 []:
    CoordinateZ DataArray_t R8 []:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [1,1]:
    ElementConnectivity DataArray_t {dtype} []:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [0,0,0]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0,0,0]:
    Cell DataArray_t {dtype} [0,0,0]:
"""
  dzones = parse_yaml_cgns.to_nodes(dt)

  keep_alive = []

  mpart = PDM.MultiPart(2, np.array([1,2], dtype=np.int32), 0, 1, 1, None, comm)
  partU.set_mpart_dmeshes(mpart, dzones, comm, keep_alive)
  assert len(keep_alive) == len(dzones)

