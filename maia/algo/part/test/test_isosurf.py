import pytest
import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.part import isosurf as ISO

from maia.utils import test_utils as TU

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


def test_copy_referenced_families():
  source_base = PT.yaml.to_node(
  """
  Base CGNSBase_t:
    Toto Family_t:
    Tata Family_t:
    Titi Family_t:
  """)
  target_base = PT.yaml.to_node(
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


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_exchange_field_one_domain(from_api, comm):
  if comm.Get_rank() == 0:
    yt_vol = f"""
    VolZone.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        ParentElements DataArray_t:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [1,3,5,7]:
      ZoneBC ZoneBC_t:
        Zmin BC_t:
          GridLocation GridLocation_t "FaceCenter":
          PointList    IndexArray_t {dtype} [[1,2,3,4]]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [60., 40, 20, 50, 30, 10]:
      DDCell DiscreteData_t:
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
    VolZone.P0.N0 Zone_t:
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
    VolZone.P1.N0 Zone_t:
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
      ZoneType ZoneType_t "Unstructured":
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        ParentElements DataArray_t:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [2,4,6,8]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [70., 80]:
      DDCell DiscreteData_t:
        GridLocation GridLocation_t "CellCenter":
        fieldA DataArray_t [50.]:
        fieldB DataArray_t [500.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [5]:
        Vertex DataArray_t {dtype} [7,8]:
    """

  if comm.Get_rank() == 0:
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
    iso_tree  = PT.yaml.to_cgns_tree(yt_surf)
    vol_tree  = PT.yaml.to_cgns_tree(yt_vol)
    ISO._exchange_field(vol_tree, iso_tree, ["DDCell", "FSolVtx", "FSolBC"], comm)
    iso_zone = PT.get_all_Zone_t(iso_tree)[0]
  else:
    iso_zone  = PT.yaml.to_node(yt_surf)
    vol_zones = PT.yaml.to_nodes(yt_vol)
    ISO.exchange_field_one_domain(vol_zones, iso_zone, ["DDCell", "FSolVtx", "FSolBC"], comm)

  assert PT.Container.GridLocation(PT.get_node_from_name(iso_zone, "DDCell")) == "CellCenter"
  assert PT.Container.GridLocation(PT.get_node_from_name(iso_zone, "FSolVtx"))  == "Vertex"
  assert PT.get_label(PT.get_node_from_name(iso_zone, "DDCell")) == "DiscreteData_t"
  assert PT.get_label(PT.get_node_from_name(iso_zone, "FSolVtx"))  == "FlowSolution_t"
  assert np.array_equal(PT.get_node_from_path(iso_zone, "DDCell/fieldA")[1], expected_A)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "DDCell/fieldB")[1], expected_B)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolVtx/fieldC")[1], expected_C)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolBC/fieldD")[1], expected_D)
  

@pytest_parallel.mark.parallel(3)
@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
def test_exchange_empty_field(comm):
  # A reproducer for #214: we had a crash if partial containers (as ZSR) are not 
  # know by every procs *and* some arrays are not of kind R8
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  bc = PT.find_node_from_name(zone, 'Xmax')
  distri = PT.get_np_value(MT.find_Distribution(bc, 'Index'))
  dn_elt = distri[1] - distri[0]
  zsr = PT.new_ZoneSubRegion('ZSR', bc_name='Xmax', fields={'One': np.ones(dn_elt), 
                                                            'Two': 2*np.ones(dn_elt, float)}, parent=zone)
  ptree = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=True, data_transfer='ALL')
  stree = maia.algo.part.plane_slice(ptree, [1,0,0,0.9032], comm, ['ZSR'])

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(2)
def test_isosurf_one_domain(comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_zones = PT.get_all_Zone_t(part_tree)
  iso_zone = ISO.iso_surface_one_domain(part_zones, "PLANE", [1,0,0,0.25], "TRI_3", "hilbert", comm)

  assert PT.Zone.n_cell(iso_zone) == 16 and PT.Zone.n_vtx(iso_zone) == 15
  assert (PT.get_node_from_name(iso_zone, 'CoordinateX')[1] == 0.25).all()
  assert (PT.get_child_from_predicates(iso_zone, 'TRI_3/ElementRange')[1] == np.array([ 1, 16], dtype=np.int32)).all()
  assert (PT.get_child_from_predicates(iso_zone, 'BAR_2/ElementRange')[1] == np.array([17, 24], dtype=np.int32)).all()

  assert PT.get_label(PT.get_child_from_name(iso_zone, "maia#surface_data")) == 'UserDefinedData_t'

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(2)
def test_compute_elliptical_slice(comm):
  
  dist_tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, preserve_orientation=True)
  slice_tree = ISO.elliptical_slice(part_tree, [0.5,0.5,0.5,.5,1.,1.,.25**2], \
      comm, elt_type='NGON_n')
  assert maia.pytree.get_node_from_name(slice_tree, "FlowSolution") is None
  iso_zone = PT.get_all_Zone_t(slice_tree)[0]
  assert comm.allreduce(PT.Zone.n_cell(iso_zone), MPI.SUM) == 88
  
@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(1)  
def test_compute_spherical_slice(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, preserve_orientation=True)
  
  zone      = PT.get_node_from_label(part_tree, "Zone_t")
  vol_rank  = comm.Get_rank() * np.ones(PT.Zone.n_cell(zone))
  src_sol   = PT.new_FlowSolution('FlowSolution', loc='CellCenter', fields={'i_rank' : vol_rank}, parent=zone)
  slice_tree = maia.algo.part.spherical_slice(part_tree, [0.5,0.5,0.5,0.25], comm, \
      ["FlowSolution"])

  iso_zone = PT.get_all_Zone_t(slice_tree)[0]
  assert PT.Zone.n_cell(iso_zone) == 1008 and PT.Zone.n_vtx(iso_zone) == 506
  elts = PT.get_nodes_from_label(iso_zone, 'Elements_t')
  assert len(elts) == 1 and PT.Element.Type(elts[0]) == 'TRI_3'
  assert maia.pytree.get_child_from_name(iso_zone, "FlowSolution") is not None
  assert (PT.get_node_from_name(iso_zone, 'i_rank')[1] == 0).all()

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(2) 
def test_compute_plane_slice(comm):
  dist_tree = maia.factory.generate_dist_block(5, 'Poly', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, preserve_orientation=True)
  slice_tree = maia.algo.part.plane_slice(part_tree, [0,0,1,0.1], comm, elt_type='QUAD_4')
  
  iso_zone = PT.get_all_Zone_t(slice_tree)[0]
  assert PT.Zone.n_cell(iso_zone) == 32 and PT.Zone.n_vtx(iso_zone) == 45

  assert np.allclose(PT.get_node_from_name(iso_zone, 'CoordinateZ')[1], 0.1)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(1) 
def test_compute_iso_surface(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  node = PT.get_node_from_name(dist_tree, 'Zmin')
  PT.set_value(node, 'BCWall')
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, preserve_orientation=True)
  maia.algo.compute_wall_distance(part_tree, comm, point_cloud='Vertex')

  part_tree_iso = maia.algo.part.iso_surface(part_tree, "WallDistance/TurbulentDistance", iso_val=0.25,\
       containers_name=['WallDistance'], comm=comm)

  iso_zone = PT.get_all_Zone_t(part_tree_iso)[0]
  assert PT.Zone.n_cell(iso_zone) == 800 and PT.Zone.n_vtx(iso_zone) == 441

  # Iso value field should be constant
  assert np.allclose(PT.get_node_from_name(part_tree_iso, 'TurbulentDistance')[1], 0.25)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(2) 
def test_multidom(comm):
  fname = TU.mesh_dir / 'U_Naca0012_multizone.yaml'
  tree = maia.io.file_to_dist_tree(fname, comm)
  # Create a field on a single domain
  maia.algo.compute_elements_center(PT.get_all_Zone_t(tree)[2], 3, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=True, data_transfer='ALL')

  # Should work
  stree = maia.algo.part.plane_slice(ptree, [0,0,1,0.5], comm, ['Geometry_3d'])
  assert len(PT.get_all_Zone_t(stree)) == 3
  assert PT.get_node_from_name_and_label(stree, 'Geometry_3d', 'DiscreteData_t') is not None

  # Should also work
  stree = maia.algo.part.plane_slice(ptree, [0,0,1,0.5], comm, 'ALL')
  assert len(PT.get_all_Zone_t(stree)) == 3
  assert PT.get_node_from_name_and_label(stree, 'Geometry_3d', 'DiscreteData_t') is not None

  # Should not work (no domain have FlowSol container)
  with pytest.raises(ValueError):
    stree = maia.algo.part.plane_slice(ptree, [0,0,1,0.5], comm, ['Geometry_3d', 'FlowSol'])