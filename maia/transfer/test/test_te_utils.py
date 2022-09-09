import pytest
import numpy as np

import Converter.Internal as I
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype
from maia.pytree.yaml import parse_yaml_cgns
from maia.transfer import utils
from pytest_mpi_check._decorator import mark_mpi_test

def test_get_partitioned_zones():
  dt = """
BaseA CGNSBase_t:
  Zone1 Zone_t:
  Zone2.With.dot Zone_t:
  Zone3 Zone_t:
BaseB CGNSBase_t:
  Zone3 Zone_t:
"""
  pt = """
BaseA CGNSBase_t:
  Zone1.P0.N1 Zone_t:
  Zone1.P0.N2 Zone_t:
  Zone2.With.dot.P0.N0 Zone_t:
BaseB CGNSBase_t:
  Zone3.P0.N0 Zone_t:
"""
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  assert [I.getName(zone) for zone in utils.get_partitioned_zones(part_tree, 'BaseA/Zone1')]\
      == ['Zone1.P0.N1', 'Zone1.P0.N2']
  assert [I.getName(zone) for zone in utils.get_partitioned_zones(part_tree, 'BaseA/Zone2.With.dot')]\
      == ['Zone2.With.dot.P0.N0']
  assert [I.getName(zone) for zone in utils.get_partitioned_zones(part_tree, 'BaseA/Zone3')]\
      == []
  assert [I.getName(zone) for zone in utils.get_partitioned_zones(part_tree, 'BaseB/Zone3')]\
      == ['Zone3.P0.N0']

def test_get_cgns_distribution():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    bc1 BC_t "Farfield":
      PointList IndexArray_t:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t [1,4,4]:
  :CGNS#Distribution UserDefinedData_t:
    Cell DataArray_t [1,2,4]:
"""
  dist_zone = parse_yaml_cgns.to_node(yt)
  zone_distri = utils.get_cgns_distribution(dist_zone, 'Cell')
  bc_distri   = utils.get_cgns_distribution(I.getNodeFromPath(dist_zone, 'ZBC/bc1'), 'Index')
  assert zone_distri.dtype == bc_distri.dtype == npy_pdm_gnum_dtype
  assert (zone_distri == [1,2,4]).all()
  assert (bc_distri   == [1,4,4]).all()

@mark_mpi_test(2)
def test_create_all_elt_distribution(sub_comm):
  yt = """
  Hexa Elements_t:
    ElementRange IndexRange_t [61,80]:
  Quad Elements_t:
    ElementRange IndexRange_t [1,60]:
  Tetra Elements_t:
    ElementRange IndexRange_t [81,100]:
"""
  dist_elts = parse_yaml_cgns.to_nodes(yt)
  distri = utils.create_all_elt_distribution(dist_elts, sub_comm)
  assert distri.dtype == npy_pdm_gnum_dtype
  if sub_comm.Get_rank() == 0:
    assert (distri == [0,50,100]).all()
  elif sub_comm.Get_rank() == 1:
    assert (distri == [50,100,100]).all()

def test_collect_cgns_g_numering():
  yt = """
Zone.P0.N0 Zone_t:
  ZBC ZoneBC_t:
    bc1 BC_t "Farfield":
      PointList IndexArray_t:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t [1,2,3,4]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Cell DataArray_t [1,2]:
Zone.P0.N1 Zone_t:
  ZBC ZoneBC_t:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Cell DataArray_t [3,4]:
"""
  part_zones = parse_yaml_cgns.to_nodes(yt)
  cell_lngn = utils.collect_cgns_g_numbering(part_zones, 'Cell')
  bc1_lngn  = utils.collect_cgns_g_numbering(part_zones, 'Index', 'ZBC/bc1')
  assert len(cell_lngn) == len(bc1_lngn) == 2
  assert (cell_lngn[0] == [1,2]).all()
  assert (cell_lngn[1] == [3,4]).all()
  assert (bc1_lngn[0] == [1,2,3,4]).all()
  assert (bc1_lngn[1] == []).all()

def test_create_all_elt_g_numbering():
  yt = """
Zone.P0.N0 Zone_t:
  Quad Elements_t:
    ElementRange IndexRange_t [1,3]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t [6,1,4]:
  Hexa Elements_t:
    ElementRange IndexRange_t [4,4]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t [1]:
  Tetra Elements_t:
    ElementRange IndexRange_t [5,8]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t [2,1,4,3]:
Zone.P0.N1 Zone_t:
  Hexa Elements_t:
    ElementRange IndexRange_t [1,1]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t [2]:
  Quad Elements_t:
    ElementRange IndexRange_t [2,4]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t [2,3,5]:
"""
  part_zones = parse_yaml_cgns.to_nodes(yt)
  dist_elts = [I.newElements('Hexa',  erange=[7,8]),
               I.newElements('Quad',  erange=[1,6]),
               I.newElements('Tetra', erange=[9,12])]
  assert (utils.create_all_elt_g_numbering(part_zones[0], dist_elts) == \
      [6,1,4,6+1,8+2,8+1,8+4,8+3]).all()
  assert (utils.create_all_elt_g_numbering(part_zones[1], dist_elts) == \
      [2,3,5,6+2]).all()

def test_get_entities_numbering():
  zoneS = I.newZone(ztype='Structured')
  expected_vtx_lngn = np.array([4,21,1,2,8,12])
  expected_face_lngn = np.array([44,23,94,12])
  expected_cell_lngn = np.array([], int)

  gnum_arrays = {'Cell' : expected_cell_lngn, 'Vertex' : expected_vtx_lngn, 'Face' : expected_face_lngn}
  gnum_node = MT.newGlobalNumbering(gnum_arrays, zoneS)
  vtx_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneS, as_pdm=False)
  assert vtx_lngn.dtype == np.int64
  assert (cell_lngn == expected_cell_lngn).all()
  assert (face_lngn == expected_face_lngn).all()

  zoneU = I.newZone(ztype='Unstructured')
  gnum_arrays = {'Cell' : expected_cell_lngn, 'Vertex' : expected_vtx_lngn}
  gnum_node = MT.newGlobalNumbering(gnum_arrays, zoneU)

  with pytest.raises(RuntimeError):
    vtx_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)

  ngon = I.newElements(etype='NGON', parent=zoneU)
  gnum_node = MT.newGlobalNumbering({'Element' : expected_face_lngn}, ngon)
  vtx_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)
  assert vtx_lngn.dtype == npy_pdm_gnum_dtype
  assert (vtx_lngn == expected_vtx_lngn).all()
  assert (face_lngn == expected_face_lngn).all()

  ngon = I.newElements('ElementsTwo', etype='NGON', parent=zoneU)
  with pytest.raises(RuntimeError):
    vtx_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)
