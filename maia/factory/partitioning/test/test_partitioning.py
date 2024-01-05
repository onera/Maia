import pytest
import pytest_parallel
from mpi4py import MPI
import numpy as np

import maia
from maia.pytree.yaml   import parse_yaml_cgns, parse_cgns_yaml
from maia.factory import generate_dist_block
from maia.factory import partition_dist_tree
from maia.utils import par_utils, s_numbering

import maia.pytree      as PT
import maia.pytree.maia as MT

@pytest_parallel.mark.parallel(2)
class Test_split_ngon_2d:

  def get_distree(self, comm):
    dist_tree = generate_dist_block(4, "QUAD_4", comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
    return dist_tree

  def check_elts(self, part_tree, comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [2,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    edge = PT.get_node_from_name(part_tree, 'EdgeElements')
    ngon = PT.get_node_from_name(part_tree, 'NGonElements')

    assert PT.get_child_from_name(edge, 'ParentElements') is not None
    assert PT.get_child_from_name(ngon, 'ParentElements') is None
  def check_bcs(self, part_tree, comm):
    bc_ymin = PT.get_node_from_name(part_tree, 'Ymin')
    bc_ymax = PT.get_node_from_name(part_tree, 'Ymax')
    bc_xmax = PT.get_node_from_name(part_tree, 'Xmax')
    if comm.Get_rank() == 0:
      assert (PT.get_child_from_name(bc_ymin, 'PointList')[1] == [[1,2,4]]).all()
      assert (PT.get_child_from_name(bc_xmax, 'PointList')[1] == [[8]]).all()
      assert bc_ymax is None
    elif comm.Get_rank() == 1:
      assert (PT.get_child_from_name(bc_ymax, 'PointList')[1] == [[10,12,13]]).all()
      assert (PT.get_child_from_name(bc_xmax, 'PointList')[1] == [[4,11]]).all()
      assert bc_ymin is None

  @pytest.mark.parametrize("no_pe", [False, True])
  def test_input_pe(self, no_pe, comm):
    dist_tree = self.get_distree(comm)
    if no_pe:
      PT.rm_nodes_from_name(dist_tree, 'ParentElements')

    part_tree = partition_dist_tree(dist_tree, comm, graph_part_tool='hilbert')
    self.check_elts(part_tree, comm)
    self.check_bcs(part_tree, comm)

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, comm):
    dist_tree = self.get_distree(comm)
    part_tree = partition_dist_tree(dist_tree, comm, graph_part_tool='hilbert', part_interface_loc=output_jn_loc)

    self.check_elts(part_tree, comm)
    self.check_bcs(part_tree, comm)
    expected_loc = {"Vertex" : "Vertex", "FaceCenter" : "EdgeCenter"}
    for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(gc) == expected_loc[output_jn_loc]


@pytest_parallel.mark.parallel(2)
class Test_split_elt_2d:
  def get_distree(self, comm):
    return generate_dist_block(4, "QUAD_4", comm)

  def check_elts(self, part_tree, comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [2,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    bar  = PT.get_node_from_name(part_tree, 'BAR_2.0')
    quad = PT.get_node_from_name(part_tree, 'QUAD_4.0')
    if comm.Get_rank() == 0:
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [1,5]).all()
      assert (PT.get_child_from_name(bar, 'ElementRange')[1] == [6,11]).all()
    if comm.Get_rank() == 1:
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [1,4]).all()
      assert (PT.get_child_from_name(bar, 'ElementRange')[1] == [5,10]).all()
  def check_bcs(self, part_tree, comm):
    bc_ymin = PT.get_node_from_name(part_tree, 'Ymin')
    bc_ymax = PT.get_node_from_name(part_tree, 'Ymax')
    bc_xmax = PT.get_node_from_name(part_tree, 'Xmax')
    if comm.Get_rank() == 0:
      assert (PT.get_child_from_name(bc_ymin, 'PointList')[1] == [[6,7,8]]).all()
      assert (PT.get_child_from_name(bc_xmax, 'PointList')[1] == [[11]]).all()
      assert bc_ymax is None
    elif comm.Get_rank() == 1:
      assert (PT.get_child_from_name(bc_ymax, 'PointList')[1] == [[5,6,7]]).all()
      assert (PT.get_child_from_name(bc_xmax, 'PointList')[1] == [[9,10]]).all()
      assert bc_ymin is None

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, comm):
    dist_tree = self.get_distree(comm)

    if output_jn_loc == 'FaceCenter':
      with pytest.raises(NotImplementedError):
        part_tree = partition_dist_tree(dist_tree, comm, part_interface_loc=output_jn_loc)
    else:
      part_tree = partition_dist_tree(dist_tree, comm, part_interface_loc=output_jn_loc)
      for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
        assert PT.Subset.GridLocation(gc) == "Vertex"

  @pytest.mark.parametrize("output_connectivity", ["Element", "NGon"])
  def test_output_elts(self, output_connectivity, comm):
    dist_tree = self.get_distree(comm)
    part_tree = partition_dist_tree(dist_tree, comm, graph_part_tool='hilbert', output_connectivity=output_connectivity)

    if output_connectivity == 'Element':
      self.check_elts(part_tree, comm)
      self.check_bcs(part_tree, comm)
    if output_connectivity == 'NGon':
      assert PT.get_node_from_name(part_tree, 'QUAD_4*') is None
      assert PT.get_node_from_name(part_tree, 'NFaceElements') is None
      Test_split_ngon_2d.check_elts(self, part_tree, comm)
      Test_split_ngon_2d.check_bcs(self, part_tree, comm)
    
@pytest_parallel.mark.parallel(2)
class Test_split_ngon_3d:
  def get_distree(self, comm):
    return generate_dist_block(4, "Poly", comm)
  def check_elts(self, part_tree, comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [3,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    assert len(PT.get_nodes_from_label(part_tree, 'Elements_t')) == 2
    nfac = PT.get_node_from_name(part_tree, 'NFaceElements')
    ngon = PT.get_node_from_name(part_tree, 'NGonElements')

    assert PT.get_child_from_name(ngon, 'ParentElements') is not None
    # This depends on partitioning tool version so use more portable test
    # if comm.Get_rank() == 0:
      # assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [1,64]).all()
      # assert (PT.get_child_from_name(nfac, 'ElementRange')[1] == [65,78]).all()
    # elif comm.Get_rank() == 1:
      # assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [1,57]).all()
      # assert (PT.get_child_from_name(nfac, 'ElementRange')[1] == [58,70]).all()
    assert PT.get_child_from_name(ngon, 'ElementRange')[1][0] == 1
    assert PT.get_child_from_name(nfac, 'ElementRange')[1][0] == PT.get_child_from_name(ngon, 'ElementRange')[1][1] + 1
    assert comm.allreduce(PT.Element.Size(nfac), MPI.SUM) == 27
    assert comm.allreduce(PT.Element.Size(ngon), MPI.SUM) == 121

  @pytest.mark.parametrize("connectivity", ["NGon+PE", "NFace+NGon", "NFace+NGon+PE"])
  def test_input_connec(self, connectivity, comm):
    dist_tree = self.get_distree(comm)
    if "NFace" in connectivity:
      removePE = not ("PE" in connectivity)
      maia.algo.pe_to_nface(dist_tree, comm, removePE)

    part_tree = partition_dist_tree(dist_tree, comm)

    # Disttree should not be modified:
    assert (PT.get_node_from_name(dist_tree, 'ParentElements') is not None) == ("PE" in connectivity)
    self.check_elts(part_tree, comm)

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, comm):
    dist_tree = self.get_distree(comm)
    part_tree = partition_dist_tree(dist_tree, comm, part_interface_loc=output_jn_loc)

    self.check_elts(part_tree, comm)
    for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(gc) == output_jn_loc

@pytest_parallel.mark.parallel(2)
class Test_split_elt_3d:
  def get_distree(self, comm):
    return generate_dist_block(4, "HEXA_8", comm)
  def check_elts(self, part_tree, comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [3,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1

    quad = PT.get_node_from_name(part_tree, 'QUAD_4.0')
    hexa = PT.get_node_from_name(part_tree, 'HEXA_8.0')
    # This depends on partitioning tool version so use more portable test
    # if comm.Get_rank() == 0:
      # assert (PT.get_child_from_name(hexa, 'ElementRange')[1] == [1,14]).all()
      # assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [15,45]).all()
    # if comm.Get_rank() == 1:
      # assert (PT.get_child_from_name(hexa, 'ElementRange')[1] == [1,13]).all()
      # assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [14,36]).all()
    assert PT.get_child_from_name(hexa, 'ElementRange')[1][0] == 1
    assert PT.get_child_from_name(quad, 'ElementRange')[1][0] == PT.get_child_from_name(hexa, 'ElementRange')[1][1] + 1
    assert comm.allreduce(PT.Element.Size(hexa), MPI.SUM) == 27
    assert comm.allreduce(PT.Element.Size(quad), MPI.SUM) == 54

  @pytest.mark.parametrize("output_connectivity", ["Element", "NGon"])
  def test_output(self, output_connectivity, comm):
    dist_tree = self.get_distree(comm)
    part_tree = partition_dist_tree(dist_tree, comm, output_connectivity=output_connectivity)

    if output_connectivity == 'Element':
      self.check_elts(part_tree, comm)
    else:
      Test_split_ngon_3d.check_elts(self, part_tree, comm)

@pytest_parallel.mark.parallel([2,3])
def test_split_point_cloud(comm):
  dist_tree = maia.factory.generate_dist_points(13, 'Unstructured', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_zone = PT.get_all_Zone_t(part_tree)[0]
  assert PT.Zone.n_cell(part_zone) == 0
  assert comm.allreduce(PT.Zone.n_vtx(part_zone), MPI.SUM) == 13**3

PART_TOOLS = ["hilbert"]
if maia.pdm_has_ptscotch:
  PART_TOOLS.append("ptscotch")
@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("method", PART_TOOLS)
def test_split_lines(method, comm):

  # We can not generate a line directly, so we do a 1D mesh and create bar
  n_vtx = 25
  dist_tree = maia.factory.generate_dist_points([n_vtx,1,1], 'U', comm)
  dist_base = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  dist_zone = PT.get_child_from_label(dist_base, 'Zone_t')

  PT.set_value(dist_base, [1,3])
  dist_zone[1][0,1] = n_vtx - 1

  if comm.Get_rank() == 0:
    bar_ec = np.repeat(np.arange(n_vtx, dtype=dist_zone[1].dtype), 2)[1:-1] + 1
    dn_bar = n_vtx-1
  else:
    bar_ec = np.empty(0, dtype=dist_zone[1].dtype)
    dn_bar = 0

  cell_distri = PT.get_node_from_name(dist_zone, 'Cell')[1]
  cell_distri[1] = dn_bar
  cell_distri[2] = n_vtx - 1
  bar_elts = PT.new_Elements('Lines', 'BAR_2', erange=[1,n_vtx-1], econn=bar_ec, parent=dist_zone)
  MT.newDistribution({'Element' : cell_distri.copy()}, bar_elts)

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, graph_part_tool=method)

  part_base = PT.get_all_CGNSBase_t(part_tree)[0]
  part_zone = PT.get_all_Zone_t(part_tree)[0]
  assert (PT.get_value(part_base) == PT.get_value(dist_base)).all()
  assert comm.allreduce(PT.Zone.n_cell(part_zone), MPI.SUM) == PT.Zone.n_cell(dist_zone)
  assert comm.allreduce(PT.Zone.n_vtx(part_zone), MPI.SUM) == PT.Zone.n_vtx(dist_zone)+1 # One is duplicated


@pytest_parallel.mark.parallel(2)
def test_split_structured(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'Structured', comm)
  
  zone_n = PT.get_node_from_path(dist_tree, 'Base/zone')
  zonebc_n = PT.get_child_from_name(zone_n, 'ZoneBC')
  
  xmin_n = PT.get_child_from_name(zonebc_n,'Xmin')
  pr = PT.get_value(PT.get_child_from_name(xmin_n,'PointRange'))
  
  pr_1 = np.copy(pr)
  pr_1[1][1] = int(pr[1][1]-1)/2 + 1
  xmin_1 = PT.new_BC('Xmin1_wo_DS',point_range=pr_1,parent=zonebc_n)
  MT.newDistribution({'Index': par_utils.uniform_distribution(PT.Subset.n_elem(xmin_1),comm)}, xmin_1)
  
  pr_2 = np.copy(pr)
  pr_2[1][0] = int(pr[1][1]-1)/2 + 1
  xmin_2 = PT.new_BC('Xmin2_w_DS',point_range=pr_2,parent=zonebc_n)
  MT.newDistribution({'Index': par_utils.uniform_distribution(PT.Subset.n_elem(xmin_2),comm)}, xmin_2)
  bcds = PT.new_node('BCDataSet','BCDataSet_t',value='Null',parent=xmin_2)
  PT.new_GridLocation('IFaceCenter',parent=bcds)
  pr_ds = np.copy(pr_2)
  pr_ds[1][0] = pr_2[1][0]+1
  pr_ds[1][1] = pr_2[1][1]-2
  pr_ds[2][0] = pr_2[2][0]+1
  pr_ds[2][1] = pr_2[2][1]-2
  PT.new_IndexRange(value=pr_ds,parent=bcds)
  MT.newDistribution({'Index': par_utils.uniform_distribution(PT.Subset.n_elem(bcds),comm)}, bcds)
  index_ds = MT.getDistribution(bcds, 'Index')

  bcd = PT.new_node('DirichletData','BCData_t',parent=bcds)
  i_ar = np.arange(pr_ds[0,0], pr_ds[0,1]+1, dtype=np.int32)
  j_ar = np.arange(pr_ds[1,0], pr_ds[1,1]+1, dtype=np.int32).reshape(-1,1)
  k_ar = np.arange(pr_ds[2,0], pr_ds[2,1]+1, dtype=np.int32).reshape(-1,1,1)
  num_face_all = s_numbering.ijk_to_faceiIndex(i_ar, j_ar, k_ar, PT.Zone.CellSize(zone_n), PT.Zone.VertexSize(zone_n)).flatten()
  num_face = num_face_all[PT.get_value(index_ds)[0]:PT.get_value(index_ds)[1]]
  PT.new_node('LNtoGN_DataSet','DataArray_t',value=num_face,parent=bcd)
  
  PT.rm_node_from_path(dist_tree, 'Base/zone/ZoneBC/Xmin')
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(dist_tree, comm, n_part=5)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)
  
  zone = PT.get_node_from_label(part_tree, 'Zone_t') #Check only on first zone
  dist_cell_size = MT.getGlobalNumbering(zone, 'CellSize')
  dist_cell_range = MT.getGlobalNumbering(zone, 'CellRange')
  if comm.Get_rank() == 0:
    expected_range = np.array([[1,4], [1,5], [1,5]], order='F')
  elif comm.Get_rank() == 1:
    expected_range = np.array([[5,7], [1,7], [6,10]], order='F')
  assert PT.get_label(dist_cell_size) == 'DataArray_t' and (PT.get_value(dist_cell_size) == [10,10,10]).all()
  assert PT.get_label(dist_cell_range) == 'IndexRange_t' and (PT.get_value(dist_cell_range) == expected_range).all()

  bcds_n_l = PT.get_nodes_from_name(part_tree, 'BCDataSet')
  sum_size_bcds = 0
  for bcds_n in bcds_n_l:
      index_tab = PT.get_value(MT.getGlobalNumbering(bcds_n, 'Index'))
      size_bcds = PT.Subset.n_elem(bcds_n)
      assert size_bcds == index_tab.shape[0]
      sum_size_bcds += size_bcds
      assert np.all(1 <= index_tab) and np.all(index_tab <= 24)
  
  assert comm.allreduce(sum_size_bcds, MPI.SUM) == 24

@pytest_parallel.mark.parallel(2)
def test_split_structured_2d(comm):
  dist_tree = maia.factory.generate_dist_block([11,6,1], 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_zone = PT.get_all_Zone_t(part_tree)[0]
  
  assert (PT.Zone.CellSize(part_zone) == [5,5]).all()
  assert MT.getGlobalNumbering(part_zone, 'Cell') is not None
  assert MT.getGlobalNumbering(part_zone, 'Face') is None


@pytest_parallel.mark.parallel(2)
def test_split_structured_1d(comm):
  dist_tree = maia.factory.generate_dist_block([10,1,1], 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_zone = PT.get_all_Zone_t(part_tree)[0]
  if comm.Get_rank() == 0:
    assert PT.Zone.CellSize(part_zone) == [5]
    assert (MT.getGlobalNumbering(part_zone, 'Cell')[1] == [1,2,3,4,5]).all()
  elif comm.Get_rank() == 1:
    assert PT.Zone.CellSize(part_zone) == [4]
    assert (MT.getGlobalNumbering(part_zone, 'Cell')[1] == [6,7,8,9]).all()
  assert MT.getGlobalNumbering(part_zone, 'Face') is None
  assert len(PT.get_nodes_from_label(part_zone, 'GridConnectivity1to1_t'))