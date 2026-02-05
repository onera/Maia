import pytest
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype
from maia.transfer import utils
import pytest_parallel

@pytest_parallel.mark.parallel(2)
def test_create_all_elt_distribution(comm):
  yt = """
  Hexa Elements_t:
    ElementRange IndexRange_t [61,80]:
  Quad Elements_t:
    ElementRange IndexRange_t [1,60]:
  Tetra Elements_t:
    ElementRange IndexRange_t [81,100]:
"""
  dist_elts = PT.yaml.to_nodes(yt)
  distri = utils.create_all_elt_distribution(dist_elts, comm)
  assert distri.dtype == npy_pdm_gnum_dtype
  if comm.Get_rank() == 0:
    assert (distri == [0,50,100]).all()
  elif comm.Get_rank() == 1:
    assert (distri == [50,100,100]).all()

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
  part_zones = PT.yaml.to_nodes(yt)
  dist_elts = [PT.new_Elements('Hexa',  erange=[7,8]),
               PT.new_Elements('Quad',  erange=[1,6]),
               PT.new_Elements('Tetra', erange=[9,12])]
  assert (utils.create_all_elt_g_numbering(part_zones[0], dist_elts) == \
      [6,1,4,6+1,8+2,8+1,8+4,8+3]).all()
  assert (utils.create_all_elt_g_numbering(part_zones[1], dist_elts) == \
      [2,3,5,6+2]).all()

def test_get_entities_numbering():
  zoneS = PT.new_Zone(type='Structured')
  expected_vtx_lngn = np.array([4,21,1,2,8,12])
  expected_face_lngn = np.array([44,23,94,12])
  expected_cell_lngn = np.array([], int)

  gnum_arrays = {'Cell' : expected_cell_lngn, 'Vertex' : expected_vtx_lngn, 'Face' : expected_face_lngn}
  gnum_node = MT.new_GlobalNumbering(gnum_arrays, zoneS)
  vtx_lngn, edge_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneS)
  assert (cell_lngn == expected_cell_lngn).all()
  assert (face_lngn == expected_face_lngn).all()
  assert edge_lngn is None

  zoneU = PT.new_Zone(type='Unstructured')
  gnum_arrays = {'Cell' : expected_cell_lngn, 'Vertex' : expected_vtx_lngn}
  gnum_node = MT.new_GlobalNumbering(gnum_arrays, zoneU)

  vtx_lngn, edge_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)
  assert face_lngn is None

  ngon = PT.new_Elements(type='NGON_n', parent=zoneU)
  gnum_node = MT.new_GlobalNumbering({'Element' : expected_face_lngn}, ngon)
  edge = PT.new_Elements('EdgeElements', type='BAR_2', parent=zoneU)
  gnum_node = MT.new_GlobalNumbering({'Element' : np.array([1,6,3,2])}, edge)
  vtx_lngn, edge_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)
  assert (vtx_lngn == expected_vtx_lngn).all()
  assert (edge_lngn == [1,6,3,2]).all()
  assert (face_lngn == expected_face_lngn).all()

  ngon = PT.new_Elements('ElementsTwo', type='NGON_n', parent=zoneU)
  with pytest.raises(RuntimeError):
    vtx_lngn, edge_lngn, face_lngn, cell_lngn = utils.get_entities_numbering(zoneU)
