import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia             import npy_pdm_gnum_dtype as pdm_dtype
from maia.algo.dist   import ngon_from_std_elements as GNG

def test_raise_if_possible_overflow():
  GNG.raise_if_possible_overflow(2000000000,1) #OK
  GNG.raise_if_possible_overflow(8000000000,4) #OK
  with pytest.raises(OverflowError):
    GNG.raise_if_possible_overflow(8000000000,2) #Overflow

@pytest_parallel.mark.parallel(2)
class Test_compute_ngon_from_std_elements:

  def test_3d_mesh(self, comm):
    #Generated from G.cartTetra((0,0,0), (1./3, 1./2, 0), (3,3,2))
    rank = comm.Get_rank()
    if comm.Get_rank() == 0:
      tetra_ec = [1,2,4,10,2,5,4,14,4,10,14,13,2,10,11,14,2,4,10,14,2,6,5,14,2,12,3,6,14,15,12,6,12,14,2,11,12,2,14,6]
    elif comm.Get_rank() == 1:
      tetra_ec = [4,8,7,16,4,14,5,8,16,17,14,8,14,16,4,13,14,4,16,8,5,6,8,14,6,9,8,18,8,14,18,17,6,14,15,18,6,8,14,18]
    tetra_ec = np.array(tetra_ec, pdm_dtype)

    dist_tree = PT.new_CGNSTree()
    dist_base = PT.new_CGNSBase('Base', cell_dim=3, phy_dim=3, parent=dist_tree)
    dist_zone = PT.new_Zone('Zone', size=[[0,20,0]], type='Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : [9*rank,9*(rank+1),18], 'Cell' : [10*rank, 10*(rank+1), 20]}, dist_zone)
    tetra = PT.new_Elements('Tetra', 'TETRA_4', erange=[1,20], econn=tetra_ec, parent=dist_zone)
    MT.newDistribution({'Element' : np.array([10*rank,10*(rank+1),20], pdm_dtype)}, tetra)

    GNG.generate_ngon_from_std_elements(dist_tree, comm)

    assert PT.get_node_from_path(dist_tree, 'Base/Zone/Tetra') is None
    ngon  = PT.get_node_from_path(dist_tree, 'Base/Zone/NGonElements')
    nface = PT.get_node_from_path(dist_tree, 'Base/Zone/NFaceElements')

    assert (PT.Element.Range(ngon) == [1,56]).all()
    assert (PT.Element.Range(nface) == [57,76]).all()
    if rank == 0:
      assert (PT.get_value(MT.getDistribution(ngon, 'Element')) == [0,27,56]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'ElementConnectivity')) == [0,81,168]).all()
      assert (PT.get_value(MT.getDistribution(nface, 'Element')) == [0,10,20]).all()
      assert (PT.get_child_from_name(ngon, 'ElementConnectivity')[1][:6] == [1,4,2,2,4,5]).all()
      assert (PT.get_child_from_name(ngon, 'ParentElements')[1][8:12] == [[68,0], [67,0], [72,0], [58,61]]).all()
    elif rank == 1:
      assert (PT.get_value(MT.getDistribution(ngon, 'Element')) == [27,56,56]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'ElementConnectivity')) == [81,168,168]).all()
      assert (PT.get_value(MT.getDistribution(nface, 'Element')) == [10,20,20]).all()
      assert (PT.get_child_from_name(ngon, 'ElementConnectivity')[1][-6:] == [14,15,18,14,18,17]).all()
      assert (PT.get_child_from_name(ngon, 'ParentElements')[1][4:8] == [[59,70], [67,0], [64,66], [73,76]]).all()
    assert (PT.get_child_from_name(nface, 'ElementStartOffset')[1] ==  np.arange(0,44,4)+40*rank).all()

  def test_2d_mesh(self, comm):
    #Generated from G.cartHexa((0,0,0), (1./3, 1./2, 0), (4,3,1))
    rank = comm.Get_rank()
    if comm.Get_rank() == 0:
      quad_ec = np.array([1,2,6,5,2,3,7,6,3,4,8,7], pdm_dtype)
    elif comm.Get_rank() == 1:
      quad_ec = np.array([5,6,10,9,6,7,11,10,7,8,12,11], pdm_dtype)

    dist_tree = PT.new_CGNSTree()
    dist_base = PT.new_CGNSBase('Base', cell_dim=2, phy_dim=3, parent=dist_tree)
    dist_zone = PT.new_Zone('Zone', size=[[12,6,0]], type='Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : np.array([6*rank,6*(rank+1),12], pdm_dtype),
                        'Cell' : np.array([3*rank, 3*(rank+1), 6], pdm_dtype)}, dist_zone)
    quad = PT.new_Elements('Quad', 'QUAD_4', erange=[1,6], econn=quad_ec, parent=dist_zone)
    MT.newDistribution({'Element' : np.array([3*rank,3*(rank+1),6], pdm_dtype)}, quad)

    GNG.generate_ngon_from_std_elements(dist_tree, comm)

    assert PT.get_node_from_path(dist_tree, 'Base/Zone/Quad') is None
    edge  = PT.get_node_from_path(dist_tree, 'Base/Zone/EdgeElements')
    ngon  = PT.get_node_from_path(dist_tree, 'Base/Zone/NGonElements')
    assert PT.get_node_from_path(dist_tree, 'Base/Zone/NFaceElements') is None

    assert (PT.Element.Range(edge) == [1,17]).all()
    assert (PT.Element.Range(ngon) == [18,23]).all()
    if rank == 0:
      assert (PT.get_value(MT.getDistribution(edge, 'Element')) == [0,9,17]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'Element')) == [0,3,6]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'ElementConnectivity')) == [0,12,24]).all()

      assert (PT.get_child_from_name(edge, 'ElementConnectivity')[1] == [1,2,2,3,5,1,3,4,2,6,3,7,6,5,4,8,7,6]).all()
      assert (PT.get_child_from_name(ngon, 'ElementStartOffset')[1] == [0,4,8,12]).all()
      assert (PT.get_child_from_name(ngon, 'ElementConnectivity')[1] == [1,2,6,5, 6,2,3,7, 7,3,4,8]).all()
    elif rank == 1:
      assert (PT.get_value(MT.getDistribution(edge, 'Element')) == [9,17,17]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'Element')) == [3,6,6]).all()
      assert (PT.get_value(MT.getDistribution(ngon, 'ElementConnectivity')) == [12,24,24]).all()

      assert (PT.get_child_from_name(edge, 'ElementConnectivity')[1] == [9,5,8,7,6,10,7,11,10,9,8,12,11,10,12,11]).all()
      assert (PT.get_child_from_name(ngon, 'ElementStartOffset')[1] == [12,16,20,24]).all()
      assert (PT.get_child_from_name(ngon, 'ElementConnectivity')[1] == [5,6,10,9, 10,6,7,11, 11,7,8,12]).all()

  def test_2d_mesh_with_bc(self, comm):
    rank = comm.Get_rank()
    if comm.Get_rank() == 0:
      quad_ec = np.array([1,2,6,5,2,3,7,6,3,4,8,7], pdm_dtype)
      bar_ec  = np.array([10,9,11,10,12,11,5,1,9,5], pdm_dtype)
    elif comm.Get_rank() == 1:
      quad_ec = np.array([5,6,10,9,6,7,11,10,7,8,12,11], pdm_dtype)
      bar_ec  = np.empty(0, dtype=pdm_dtype)

    dist_tree = PT.new_CGNSTree()
    dist_base = PT.new_CGNSBase('Base', cell_dim=2, phy_dim=3, parent=dist_tree)
    dist_zone = PT.new_Zone('Zone', size=[[12,6,0]], type='Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : [6*rank,6*(rank+1),12], 'Cell' : [3*rank, 3*(rank+1), 6]}, dist_zone)
    quad = PT.new_Elements('Quad', 'QUAD_4', erange=[1,6], econn=quad_ec, parent=dist_zone)
    MT.newDistribution({'Element' : np.array([3*rank,3*(rank+1),6], pdm_dtype)}, quad)
    bar  = PT.new_Elements('Bar', 'BAR_2', erange=[7,11], econn=bar_ec, parent=dist_zone)
    MT.newDistribution({'Element' : np.array([0,5*(1-rank),5], pdm_dtype)}, bar)
    dist_zbc  = PT.new_ZoneBC(dist_zone)
    if rank == 0:
      bca       = PT.new_BC('bcA', point_range = [[7,9]], parent=dist_zbc)
      MT.newDistribution({'Index' : [0,2,3]}, bca)
      bcb       = PT.new_BC('bcB', point_list = [[10]], parent=dist_zbc)
      MT.newDistribution({'Index' : [0,1,2]}, bcb)
      bcc       = PT.new_BC('bcC', point_range = [[1,5]], parent=dist_zbc)
      MT.newDistribution({'Index' : [0,3,5]}, bcb)
    elif rank == 1:
      bca       = PT.new_BC('bcA', point_range = [[7,9]], parent=dist_zbc)
      MT.newDistribution({'Index' : [2,3,3]}, bca)
      bcb       = PT.new_BC('bcB', point_list = [[11]], parent=dist_zbc)
      MT.newDistribution({'Index' : [1,2,2]}, bcb)
      bcc       = PT.new_BC('bcC', point_range = [[1,5]], parent=dist_zbc)
      MT.newDistribution({'Index' : [3,5,5]}, bcb)
    PT.new_GridLocation('EdgeCenter', bca)
    PT.new_GridLocation('EdgeCenter', bcb)
    PT.new_GridLocation('Vertex', bcc)

    GNG.generate_ngon_from_std_elements(dist_tree, comm)

    assert PT.get_node_from_path(dist_tree, 'Base/Zone/Quad') is None
    assert PT.get_node_from_path(dist_tree, 'Base/Zone/Bar') is None
    ngon  = PT.get_node_from_path(dist_tree, 'Base/Zone/NGonElements')
    edge = PT.get_node_from_path(dist_tree, 'Base/Zone/EdgeElements')

    #Apparently addition elements causes the distribution to change // dont retest here
    assert (PT.Element.Range(edge) == [1,17]).all()
    assert (PT.Element.Range(ngon) == [18,23]).all()
    assert PT.Subset.GridLocation(bca) == 'EdgeCenter'
    if rank == 0:
      assert (PT.get_child_from_name(bca, 'PointList')[1] == [14,16]).all()
      assert (PT.get_child_from_name(bcb, 'PointList')[1] == [3]).all()
    elif rank == 1:
      assert (PT.get_child_from_name(bca, 'PointList')[1] == [17]).all()
      assert (PT.get_child_from_name(bcb, 'PointList')[1] == [10]).all()
    assert PT.Subset.GridLocation(bcc) == 'Vertex' #Vertex bc should not have changed
    assert (PT.get_child_from_name(bcc, 'PointRange')[1] == [[1,5]]).all()

