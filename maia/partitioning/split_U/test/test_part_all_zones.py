from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I
import numpy as np
from mpi4py import MPI

import Pypdm.Pypdm as PDM

from maia                      import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils                import parse_yaml_cgns
from maia.partitioning.split_U import part_all_zones as partU

def test_prepare_part_weight():
  zones = [I.newZone('ZoneA', ztype='Unstructured'),
           I.newZone('ZoneB', ztype='Unstructured'),
           I.newZone('ZoneC', ztype='Unstructured')]
  n_part_per_zone = np.array([1,0,3], dtype=np.int32)
  d_zone_to_parts = {'ZoneA' : [.3], 'ZoneB' : [], 'ZoneC' : [.2,.5,.3]}
  assert (partU.prepare_part_weight(zones, n_part_per_zone, d_zone_to_parts)\
      == [.3,.2,.5,.3]).all()

def test_set_mpart_reordering():
  keep_alive = []
  reorder_options = {'cell_renum_method' : 'CUTHILL', 'face_renum_method' : 'LEXICOGRAPHIC'}
  n_part_per_zones = np.array([1,2], dtype=np.int32)
  mpart = PDM.MultiPart(2, np.array([1,2], dtype=np.int32), 0, 1, 1, None, MPI.COMM_SELF)
  partU.set_mpart_reordering(mpart, reorder_options, keep_alive)

@mark_mpi_test(2)
def test_set_mpart_dmeshes(sub_comm):
  dtype = 'I4' if pdm_dtype == np.int32 else 'I8'
  dt = f"""
ZoneA Zone_t:
  NGonElements Elements_t [22,0]:
    ElementConnectivity DataArray_t {dtype} []:
    ElementStartOffset DataArray_t {dtype} [0]:
    ParentElements DataArray_t {dtype} []:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [0,0,0]:
      ElementConnectivity DataArray_t {dtype} [0,0,0]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0,0,0]:
    Cell DataArray_t {dtype} [0,0,0]:
ZoneB Zone_t:
  Hexa Elements_t [17,0]:
    ElementRange ElementRange_t [1,1]:
    ElementConnectivity DataArray_t {dtype} []:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [0,0,0]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0,0,0]:
    Cell DataArray_t {dtype} [0,0,0]:
"""
  dzones = parse_yaml_cgns.to_nodes(dt)

  keep_alive = []

  mpart = PDM.MultiPart(2, np.array([1,2], dtype=np.int32), 0, 1, 1, None, sub_comm)
  partU.set_mpart_dmeshes(mpart, dzones, sub_comm, keep_alive)
  assert len(keep_alive) == len(dzones)

