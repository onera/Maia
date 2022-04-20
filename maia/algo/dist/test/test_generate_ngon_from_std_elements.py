import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia             import npy_pdm_gnum_dtype as pdm_dtype
from maia.algo.dist   import ngon_from_std_elements as GNG

@mark_mpi_test(2)
class Test_compute_ngon_from_std_elements:

  def test_3d_mesh(self, sub_comm):
    #Generated from G.cartTetra((0,0,0), (1./3, 1./2, 0), (3,3,2))
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      tetra_ec = [1,2,4,10,2,5,4,14,4,10,14,13,2,10,11,14,2,4,10,14,2,6,5,14,2,12,3,6,14,15,12,6,12,14,2,11,12,2,14,6]
    elif sub_comm.Get_rank() == 1:
      tetra_ec = [4,8,7,16,4,14,5,8,16,17,14,8,14,16,4,13,14,4,16,8,5,6,8,14,6,9,8,18,8,14,18,17,6,14,15,18,6,8,14,18]
    tetra_ec = np.array(tetra_ec, pdm_dtype)

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 3, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [0,20,0], 'Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : [9*rank,9*(rank+1),18], 'Cell' : [10*rank, 10*(rank+1), 20]}, dist_zone)
    tetra = I.newElements('Tetra', 'TETRA', tetra_ec, [1,20], parent=dist_zone)
    MT.newDistribution({'Element' : np.array([10*rank,10*(rank+1),20], pdm_dtype)}, tetra)

    GNG.compute_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Tetra') == tetra
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    assert (PT.Element.Range(ngon) == [21,76]).all()
    assert (PT.Element.Range(nface) == [77,96]).all()
    if rank == 0:
      assert (I.getVal(MT.getDistribution(ngon, 'Element')) == [0,27,56]).all()
      assert (I.getVal(MT.getDistribution(ngon, 'ElementConnectivity')) == [0,81,168]).all()
      assert (I.getVal(MT.getDistribution(nface, 'Element')) == [0,10,20]).all()
      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1][:6] == [1,4,2,2,4,5]).all()
      assert (I.getNodeFromName(ngon, 'ParentElements')[1][8:12] == [[88,0], [87,0], [92,0], [78,81]]).all()
    elif rank == 1:
      assert (I.getVal(MT.getDistribution(ngon, 'Element')) == [27,56,56]).all()
      assert (I.getVal(MT.getDistribution(ngon, 'ElementConnectivity')) == [81,168,168]).all()
      assert (I.getVal(MT.getDistribution(nface, 'Element')) == [10,20,20]).all()
      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1][-6:] == [14,15,18,14,18,17]).all()
      assert (I.getNodeFromName(ngon, 'ParentElements')[1][4:8] == [[79,90], [87,0], [84,86], [93,96]]).all()
    assert (I.getNodeFromName(nface, 'ElementStartOffset')[1] ==  np.arange(0,44,4)+40*rank).all()

  def test_2d_mesh(self, sub_comm):
    #Generated from G.cartHexa((0,0,0), (1./3, 1./2, 0), (4,3,1))
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      quad_ec = np.array([1,2,6,5,2,3,7,6,3,4,8,7], pdm_dtype)
    elif sub_comm.Get_rank() == 1:
      quad_ec = np.array([5,6,10,9,6,7,11,10,7,8,12,11], pdm_dtype)

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 2, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [12,6,0], 'Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : np.array([6*rank,6*(rank+1),12], pdm_dtype),
                        'Cell' : np.array([3*rank, 3*(rank+1), 6], pdm_dtype)}, dist_zone)
    quad = I.newElements('Quad', 'QUAD', quad_ec, [1,6], parent=dist_zone)
    MT.newDistribution({'Element' : np.array([3*rank,3*(rank+1),6], pdm_dtype)}, quad)

    GNG.compute_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Quad') == quad
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    assert (PT.Element.Range(ngon) == [7,23]).all()
    assert (PT.Element.Range(nface) == [24,29]).all()
    if rank == 0:
      assert (I.getVal(MT.getDistribution(ngon, 'Element')) == [0,9,17]).all()
      assert (I.getVal(MT.getDistribution(nface, 'Element')) == [0,3,6]).all()
      assert (I.getVal(MT.getDistribution(nface, 'ElementConnectivity')) == [0,12,24]).all()

      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == [1,2,2,3,5,1,3,4,2,6,3,7,6,5,4,8,7,6]).all()
      assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == [0,2,4,6,8,10,12,14,16,18]).all()
      assert (I.getNodeFromName(nface, 'ElementConnectivity')[1] == [1,3,5,7,5,2,6,9,6,4,8,11]).all()
    elif rank == 1:
      assert (I.getVal(MT.getDistribution(ngon, 'Element')) == [9,17,17]).all()
      assert (I.getVal(MT.getDistribution(nface, 'Element')) == [3,6,6]).all()
      assert (I.getVal(MT.getDistribution(nface, 'ElementConnectivity')) == [12,24,24]).all()

      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == [9,5,8,7,6,10,7,11,10,9,8,12,11,10,12,11]).all()
      assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == [18,20,22,24,26,28,30,32,34]).all()
      assert (I.getNodeFromName(nface, 'ElementConnectivity')[1] == [7,10,12,14,12,9,13,16,13,11,15,17]).all()

  def test_2d_mesh_with_bc(self, sub_comm):
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      quad_ec = np.array([1,2,6,5,2,3,7,6,3,4,8,7], pdm_dtype)
      bar_ec  = np.array([10,9,11,10,12,11,5,1,9,5], pdm_dtype)
    elif sub_comm.Get_rank() == 1:
      quad_ec = np.array([5,6,10,9,6,7,11,10,7,8,12,11], pdm_dtype)
      bar_ec  = np.empty(0, dtype=pdm_dtype)

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 2, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [12,6,0], 'Unstructured', parent=dist_base)
    MT.newDistribution({'Vertex' : [6*rank,6*(rank+1),12], 'Cell' : [3*rank, 3*(rank+1), 6]}, dist_zone)
    quad = I.newElements('Quad', 'QUAD', quad_ec, [1,6], parent=dist_zone)
    MT.newDistribution({'Element' : np.array([3*rank,3*(rank+1),6], pdm_dtype)}, quad)
    bar  = I.newElements('Bar', 'BAR', bar_ec, [7,11], parent=dist_zone)
    MT.newDistribution({'Element' : np.array([0,5*(1-rank),5], pdm_dtype)}, bar)
    dist_zbc  = I.newZoneBC(dist_zone)
    if rank == 0:
      bca       = I.newBC('bcA', pointRange = [[7,9]], parent=dist_zbc)
      MT.newDistribution({'Index' : [0,2,3]}, bca)
      bcb       = I.newBC('bcB', pointList = [[10]], parent=dist_zbc)
      MT.newDistribution({'Index' : [0,1,2]}, bcb)
    elif rank == 1:
      bca       = I.newBC('bcA', pointRange = [[7,9]], parent=dist_zbc)
      MT.newDistribution({'Index' : [2,3,3]}, bca)
      bcb       = I.newBC('bcB', pointList = [[11]], parent=dist_zbc)
      MT.newDistribution({'Index' : [1,2,2]}, bcb)
    I.newGridLocation('EdgeCenter', bca)
    I.newGridLocation('EdgeCenter', bcb)

    GNG.compute_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Quad') == quad
    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Bar') == bar
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    #Apparently addition elements causes the distribution to change // dont retest here
    assert (PT.Element.Range(ngon) == [12,28]).all()
    assert (PT.Element.Range(nface) == [29,34]).all()
    assert PT.Subset.GridLocation(bca) == 'EdgeCenter'
    if rank == 0:
      assert (I.getNodeFromName(bca, 'PointList')[1] == [25,27]).all()
      assert (I.getNodeFromName(bcb, 'PointList')[1] == [14]).all()
    elif rank == 1:
      assert (I.getNodeFromName(bca, 'PointList')[1] == [28]).all()
      assert (I.getNodeFromName(bcb, 'PointList')[1] == [21]).all()

@mark_mpi_test(2)
def test_generate_ngon_from_std_elements_jc(sub_comm):
  #Generated from G.cartTetra((0,0,0), (1./3, 1./2, 0), (3,3,2))
  rank = sub_comm.Get_rank()
  if sub_comm.Get_rank() == 0:
    tetra_ec = [1,2,4,10,2,5,4,14,4,10,14,13,2,10,11,14,2,4,10,14,2,6,5,14,2,12,3,6,14,15,12,6,12,14,2,11,12,2,14,6]
  elif sub_comm.Get_rank() == 1:
    tetra_ec = [4,8,7,16,4,14,5,8,16,17,14,8,14,16,4,13,14,4,16,8,5,6,8,14,6,9,8,18,8,14,18,17,6,14,15,18,6,8,14,18]

  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase('Base', 3, 3, parent=dist_tree)
  dist_zone = I.newZone('Zone', [0,20,0], 'Unstructured', parent=dist_base)
  MT.newDistribution({'Vertex' : [9*rank,9*(rank+1),18], 'Cell' : [10*rank, 10*(rank+1), 20]}, dist_zone)
  tetra = I.newElements('Tetra', 'TETRA', np.array(tetra_ec, pdm_dtype), [1,20], parent=dist_zone)
  MT.newDistribution({'Element' : np.array([10*rank,10*(rank+1),20], pdm_dtype)}, tetra)

  GNG.generate_ngon_from_std_elements(dist_tree, sub_comm)

  assert I.getNodeFromPath(dist_tree, 'Base/Zone/Tetra') is None
  ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
  nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

  assert (PT.Element.Range(ngon) == [1,56]).all()
  assert (PT.Element.Range(nface) == [57,76]).all()
  if rank == 0:
    assert (I.getNodeFromName(ngon, 'ParentElements')[1][8:12] == [[68,0], [67,0], [72,0], [58,61]]).all()
  elif rank == 1:
    assert (I.getNodeFromName(ngon, 'ParentElements')[1][4:8] == [[59,70], [67,0], [64,66], [73,76]]).all()
