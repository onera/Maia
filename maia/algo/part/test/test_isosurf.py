import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia
import maia.pytree        as PT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia.algo.part import isosurf as ISO

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


def test_copy_referenced_families():
  source_base = parse_yaml_cgns.to_node(
  """
  Base CGNSBase_t:
    Toto Family_t:
    Tata Family_t:
    Titi Family_t:
  """)
  target_base = parse_yaml_cgns.to_node(
  """
  Base CGNSBase_t:
    Tyty Family_t: #Already in target tree
    ZoneA Zone_t:
      FamilyName FamilyName_t "Toto":
      AddFamilyName AdditionalFamilyName_t "Tutu": #Not in source tree
    ZoneB Zone_t:
      AdditionalFamilyName AdditionalFamilyName_t "Titi":
  """)
  ISO.copy_referenced_families(source_base, target_base)
  assert PT.get_child_from_name(target_base, 'Tyty') is not None
  assert PT.get_child_from_name(target_base, 'Toto') is not None
  assert PT.get_child_from_name(target_base, 'Titi') is not None
  assert PT.get_child_from_name(target_base, 'Tata') is None


@mark_mpi_test(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_exchange_field_one_domain(from_api, sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_vol = f"""
    VolZone.P0.N0 Zone_t:
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [1,3,5,7]:
      ZoneBC ZoneBC_t:
        Zmin BC_t:
          GridLocation GridLocation_t "FaceCenter":
          PointList    IndexArray_t {dtype} [[1,2,3,4]]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [60., 40, 20, 50, 30, 10]:
      FSolCell FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldA DataArray_t [40., 30., 20., 10.]:
        fieldB DataArray_t [400., 300., 200., 100.]:
      FSolBC ZoneSubRegion_t:
        BCRegionName Descriptor_t "Zmin":
        fieldD DataArray_t R8 [-1., -3., -5., -7.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [4,3,2,1]:
        Vertex DataArray_t {dtype} [6,4,2,5,3,1]:
    """
    yt_surf = f"""
    VolZone_iso.P0.N0 Zone_t:
      BAR_2 Elements_t [3,0]:
        ElementRange IndexRange_t [1,3]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [3,2]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [2]:
        Vertex DataArray_t {dtype} [1,2]:
      maia#surface_data UserDefinedData_t:
        Vtx_parent_weight DataArray_t [1., 1.]:
        Vtx_parent_gnum DataArray_t {dtype} [6,5]:
        Vtx_parent_idx DataArray_t I4 [0,1,2]:
        Cell_parent_gnum DataArray_t {dtype} [4]:
        Face_parent_bnd_edges DataArray_t {dtype} [5, 1]:
    """
  else:
    yt_surf = f"""
    VolZone_iso.P1.N0 Zone_t:
      BAR_2 Elements_t [3,0]:
        ElementRange IndexRange_t [1,3]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [1]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [1,3]:
        Vertex DataArray_t {dtype} [2,3]:
      maia#surface_data UserDefinedData_t:
        Vtx_parent_weight DataArray_t [1., .5, .5]:
        Vtx_parent_gnum DataArray_t {dtype} [5,1,2]:
        Vtx_parent_idx DataArray_t I4 [0,1,3]:
        Cell_parent_gnum DataArray_t {dtype} [3, 1]:
        Face_parent_bnd_edges DataArray_t {dtype} [3]:
    """
    yt_vol = f"""
    VolZone.P1.N0 Zone_t:
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [2,4,6,8]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [70., 80]:
      FSolCell FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldA DataArray_t [50.]:
        fieldB DataArray_t [500.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [5]:
        Vertex DataArray_t {dtype} [7,8]:
    """

  if sub_comm.Get_rank() == 0:
    expected_A = np.array([40.])
    expected_B = np.array([400.])
    expected_C = np.array([60., 50.])
    expected_D = np.array([-5., -1.])
  else:
    expected_A = np.array([30., 10.])
    expected_B = np.array([300., 100.])
    expected_C = np.array([50., 15.])
    expected_D = np.array([-3.])

  if from_api:
    iso_tree  = parse_yaml_cgns.to_cgns_tree(yt_surf)
    vol_tree  = parse_yaml_cgns.to_cgns_tree(yt_vol)
    ISO._exchange_field(vol_tree, iso_tree, ["FSolCell", "FSolVtx", "FSolBC"], sub_comm)
    iso_zone = PT.get_all_Zone_t(iso_tree)[0]
  else:
    iso_zone  = parse_yaml_cgns.to_node(yt_surf)
    vol_zones = parse_yaml_cgns.to_nodes(yt_vol)
    ISO.exchange_field_one_domain(vol_zones, iso_zone, ["FSolCell", "FSolVtx", "FSolBC"], sub_comm)
  
  assert PT.Subset.GridLocation(PT.get_node_from_name(iso_zone, "FSolCell")) == "CellCenter"
  assert PT.Subset.GridLocation(PT.get_node_from_name(iso_zone, "FSolVtx")) == "Vertex"
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolCell/fieldA")[1], expected_A)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolCell/fieldB")[1], expected_B)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolVtx/fieldC")[1], expected_C)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolBC/fieldD")[1], expected_D)
  

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@mark_mpi_test(2)
def test_isosurf_one_domain(sub_comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", sub_comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm)

  part_zones = PT.get_all_Zone_t(part_tree)
  iso_zone = ISO.iso_surface_one_domain(part_zones, "PLANE", [1,0,0,0.25], "TRI_3", sub_comm)

  assert PT.Zone.n_cell(iso_zone) == 16 and PT.Zone.n_vtx(iso_zone) == 15
  assert (PT.get_node_from_name(iso_zone, 'CoordinateX')[1] == 0.25).all()
  assert (PT.get_child_from_predicates(iso_zone, 'TRI_3/ElementRange')[1] == np.array([ 1, 16], dtype=np.int32)).all()
  assert (PT.get_child_from_predicates(iso_zone, 'BAR_2/ElementRange')[1] == np.array([17, 24], dtype=np.int32)).all()

  assert PT.get_label(PT.get_child_from_name(iso_zone, "maia#surface_data")) == 'UserDefinedData_t'

