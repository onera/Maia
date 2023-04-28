import pytest
from pytest_mpi_check._decorator import mark_mpi_test
from mpi4py import MPI

import maia
from maia.pytree.yaml   import parse_yaml_cgns, parse_cgns_yaml
from maia.factory import generate_dist_block
from maia.factory import partition_dist_tree

import maia.pytree as PT

@mark_mpi_test(2)
class Test_split_ngon_2d:

  def get_distree(self, sub_comm):
    dist_tree = generate_dist_block(4, "QUAD_4", sub_comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, sub_comm)
    return dist_tree

  def check_elts(self, part_tree, sub_comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [2,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    edge = PT.get_node_from_name(part_tree, 'EdgeElements')
    ngon = PT.get_node_from_name(part_tree, 'NGonElements')

    assert PT.get_child_from_name(edge, 'ParentElements') is not None
    assert PT.get_child_from_name(ngon, 'ParentElements') is None
    if sub_comm.Get_rank() == 0:
      assert (PT.get_child_from_name(edge, 'ElementRange')[1] == [1,15]).all()
      assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [16,20]).all()
    elif sub_comm.Get_rank() == 1:
      assert (PT.get_child_from_name(edge, 'ElementRange')[1] == [1,13]).all()
      assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [14,17]).all()

  @pytest.mark.parametrize("no_pe", [False, True])
  def test_input_pe(self, no_pe, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    if no_pe:
      PT.rm_nodes_from_name(dist_tree, 'ParentElements')

    part_tree = partition_dist_tree(dist_tree, sub_comm)
    self.check_elts(part_tree, sub_comm)

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    part_tree = partition_dist_tree(dist_tree, sub_comm, part_interface_loc=output_jn_loc)

    self.check_elts(part_tree, sub_comm)
    expected_loc = {"Vertex" : "Vertex", "FaceCenter" : "EdgeCenter"}
    for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(gc) == expected_loc[output_jn_loc]


@mark_mpi_test(2)
class Test_split_elt_2d:
  def get_distree(self, sub_comm):
    return generate_dist_block(4, "QUAD_4", sub_comm)

  def check_elts(self, part_tree, sub_comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [2,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    bar  = PT.get_node_from_name(part_tree, 'BAR_2.0')
    quad = PT.get_node_from_name(part_tree, 'QUAD_4.0')
    if sub_comm.Get_rank() == 0:
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [1,5]).all()
      assert (PT.get_child_from_name(bar, 'ElementRange')[1] == [6,11]).all()
    if sub_comm.Get_rank() == 1:
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [1,4]).all()
      assert (PT.get_child_from_name(bar, 'ElementRange')[1] == [5,10]).all()

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, sub_comm):
    dist_tree = self.get_distree(sub_comm)

    if output_jn_loc == 'FaceCenter':
      with pytest.raises(NotImplementedError):
        part_tree = partition_dist_tree(dist_tree, sub_comm, part_interface_loc=output_jn_loc)
    else:
      part_tree = partition_dist_tree(dist_tree, sub_comm, part_interface_loc=output_jn_loc)
      for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
        assert PT.Subset.GridLocation(gc) == "Vertex"

  def test_output_elts(self, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    part_tree = partition_dist_tree(dist_tree, sub_comm, output_connectivity="NGon")

    bar  = PT.get_node_from_name(part_tree, 'BAR_2.0')
    ngon = PT.get_node_from_name(part_tree, 'NGonElements')
    assert PT.get_node_from_name(part_tree, 'QUAD_4*') is None
    assert PT.get_node_from_name(part_tree, 'NFaceElements') is None

    # Rename to use previous check fct
    bar[0] = 'EdgeElements'
    Test_split_ngon_2d.check_elts(self, part_tree, sub_comm)
    
@mark_mpi_test(2)
class Test_split_ngon_3d:
  def get_distree(self, sub_comm):
    return generate_dist_block(4, "Poly", sub_comm)
  def check_elts(self, part_tree, sub_comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [3,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1
    assert len(PT.get_nodes_from_label(part_tree, 'Elements_t')) == 2
    nfac = PT.get_node_from_name(part_tree, 'NFaceElements')
    ngon = PT.get_node_from_name(part_tree, 'NGonElements')

    assert PT.get_child_from_name(ngon, 'ParentElements') is not None
    if sub_comm.Get_rank() == 0:
      assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [1,64]).all()
      assert (PT.get_child_from_name(nfac, 'ElementRange')[1] == [65,78]).all()
    elif sub_comm.Get_rank() == 1:
      assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [1,57]).all()
      assert (PT.get_child_from_name(nfac, 'ElementRange')[1] == [58,70]).all()

  @pytest.mark.parametrize("connectivity", ["NGon+PE", "NFace+NGon", "NFace+NGon+PE"])
  def test_input_connec(self, connectivity, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    if "NFace" in connectivity:
      removePE = not ("PE" in connectivity)
      maia.algo.pe_to_nface(dist_tree, sub_comm, removePE)

    part_tree = partition_dist_tree(dist_tree, sub_comm)

    # Disttree should not be modified:
    assert (PT.get_node_from_name(dist_tree, 'ParentElements') is not None) == ("PE" in connectivity)
    self.check_elts(part_tree, sub_comm)

  @pytest.mark.parametrize("output_jn_loc", ["Vertex", "FaceCenter"])
  def test_output_loc(self, output_jn_loc, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    part_tree = partition_dist_tree(dist_tree, sub_comm, part_interface_loc=output_jn_loc)

    self.check_elts(part_tree, sub_comm)
    for gc in PT.iter_nodes_from_label(part_tree, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(gc) == output_jn_loc

@mark_mpi_test(2)
class Test_split_elt_3d:
  def get_distree(self, sub_comm):
    return generate_dist_block(4, "HEXA_8", sub_comm)
  def check_elts(self, part_tree, sub_comm):
    assert (PT.get_all_CGNSBase_t(part_tree)[0][1] == [3,3]).all()
    assert len(PT.get_all_Zone_t(part_tree)) == 1

    quad = PT.get_node_from_name(part_tree, 'QUAD_4.0')
    hexa = PT.get_node_from_name(part_tree, 'HEXA_8.0')
    if sub_comm.Get_rank() == 0:
      assert (PT.get_child_from_name(hexa, 'ElementRange')[1] == [1,14]).all()
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [15,45]).all()
    if sub_comm.Get_rank() == 1:
      assert (PT.get_child_from_name(hexa, 'ElementRange')[1] == [1,13]).all()
      assert (PT.get_child_from_name(quad, 'ElementRange')[1] == [14,36]).all()

  @pytest.mark.parametrize("output_connectivity", ["Element", "NGon"])
  def test_output(self, output_connectivity, sub_comm):
    dist_tree = self.get_distree(sub_comm)
    part_tree = partition_dist_tree(dist_tree, sub_comm, output_connectivity=output_connectivity)

    if output_connectivity == 'Element':
      self.check_elts(part_tree, sub_comm)
    else:
      Test_split_ngon_3d.check_elts(self, part_tree, sub_comm)

@mark_mpi_test([2,3])
def test_split_point_cloud(sub_comm):
  dist_tree = maia.factory.generate_dist_points(13, 'Unstructured', sub_comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm)

  part_zone = PT.get_all_Zone_t(part_tree)[0]
  assert PT.Zone.n_cell(part_zone) == 0
  assert sub_comm.allreduce(PT.Zone.n_vtx(part_zone), MPI.SUM) == 13**3

