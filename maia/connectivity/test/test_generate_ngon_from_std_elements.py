import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I

from maia.sids         import sids
from maia.sids         import Internal_ext as IE
from maia.connectivity import generate_ngon_from_std_elements as GNG

@mark_mpi_test(2)
class Test_generate_ngon_from_std_elements:

  def test_3d_mesh(self, sub_comm):
    #Generated from G.cartTetra((0,0,0), (1./3, 1./2, 0), (3,3,2))
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      tetra_ec = [1,2,4,10,2,5,4,14,4,10,14,13,2,10,11,14,2,4,10,14,2,6,5,14,2,12,3,6,14,15,12,6,12,14,2,11,12,2,14,6]
    elif sub_comm.Get_rank() == 1:
      tetra_ec = [4,8,7,16,4,14,5,8,16,17,14,8,14,16,4,13,14,4,16,8,5,6,8,14,6,9,8,18,8,14,18,17,6,14,15,18,6,8,14,18]

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 3, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [0,20,0], 'Unstructured', parent=dist_base)
    IE.newDistribution({'Vertex' : [9*rank,9*(rank+1),18], 'Cell' : [10*rank, 10*(rank+1), 20]}, dist_zone)
    tetra = I.newElements('Tetra', 'TETRA', tetra_ec, [1,20], parent=dist_zone)
    IE.newDistribution({'Element' : [10*rank,10*(rank+1),20]}, tetra)

    GNG.generate_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Tetra') == tetra
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    assert (sids.ElementRange(ngon) == [21,76]).all()
    assert (sids.ElementRange(nface) == [77,96]).all()
    if rank == 0:
      assert (IE.getDistribution(ngon, 'Element') == [0,27,56]).all()
      assert (IE.getDistribution(ngon, 'ElementConnectivity') == [0,81,168]).all()
      assert (IE.getDistribution(nface, 'Element') == [0,10,20]).all()
      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1][:6] == [1,4,2,2,4,5]).all()
    elif rank == 1:
      assert (IE.getDistribution(ngon, 'Element') == [27,56,56]).all()
      assert (IE.getDistribution(ngon, 'ElementConnectivity') == [81,168,168]).all()
      assert (IE.getDistribution(nface, 'Element') == [10,20,20]).all()
      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1][-6:] == [16,14,17,14,18,17]).all()
    assert (I.getNodeFromName(nface, 'ElementStartOffset')[1] ==  np.arange(0,44,4)+40*rank).all()

  def test_2d_mesh(self, sub_comm):
    #Generated from G.cartHexa((0,0,0), (1./3, 1./2, 0), (4,3,1))
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      quad_ec = [1,2,6,5,2,3,7,6,3,4,8,7]
    elif sub_comm.Get_rank() == 1:
      quad_ec = [5,6,10,9,6,7,11,10,7,8,12,11]

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 2, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [12,6,0], 'Unstructured', parent=dist_base)
    IE.newDistribution({'Vertex' : [6*rank,6*(rank+1),12], 'Cell' : [3*rank, 3*(rank+1), 6]}, dist_zone)
    quad = I.newElements('Quad', 'QUAD', quad_ec, [1,6], parent=dist_zone)
    IE.newDistribution({'Element' : [3*rank,3*(rank+1),6]}, quad)

    GNG.generate_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Quad') == quad
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    assert (sids.ElementRange(ngon) == [7,23]).all()
    assert (sids.ElementRange(nface) == [24,29]).all()
    if rank == 0:
      assert (IE.getDistribution(ngon, 'Element') == [0,8,17]).all()
      assert (IE.getDistribution(nface, 'Element') == [0,3,6]).all()
      assert (IE.getDistribution(nface, 'ElementConnectivity') == [0,12,24]).all()

      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == [2,1,3,2,1,5,4,3,6,2,3,7,5,6,8,4]).all()
      assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == [0,2,4,6,8,10,12,14,16]).all()
      assert (I.getNodeFromName(nface, 'ElementConnectivity')[1] == [1,3,5,7,6,5,2,9,11,4,6,8]).all()
    elif rank == 1:
      assert (IE.getDistribution(ngon, 'Element') == [8,17,17]).all()
      assert (IE.getDistribution(nface, 'Element') == [3,6,6]).all()
      assert (IE.getDistribution(nface, 'ElementConnectivity') == [12,24,24]).all()

      assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == [6,7,5,9,8,7,6,10,11,7,9,10,12,8,10,11,11,12]).all()
      assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == [16,18,20,22,24,26,28,30,32,34]).all()
      assert (I.getNodeFromName(nface, 'ElementConnectivity')[1] == [12,7,10,14,9,12,13,16,13,11,15,17]).all()

  def test_2d_mesh_with_bc(self, sub_comm):
    rank = sub_comm.Get_rank()
    if sub_comm.Get_rank() == 0:
      quad_ec = [1,2,6,5,2,3,7,6,3,4,8,7]
      bar_ec  = [10,9,11,10,12,11,5,1,9,5]
    elif sub_comm.Get_rank() == 1:
      quad_ec = [5,6,10,9,6,7,11,10,7,8,12,11]
      bar_ec  = np.empty(0, dtype=np.int32)

    dist_tree = I.newCGNSTree()
    dist_base = I.newCGNSBase('Base', 2, 3, parent=dist_tree)
    dist_zone = I.newZone('Zone', [12,6,0], 'Unstructured', parent=dist_base)
    IE.newDistribution({'Vertex' : [6*rank,6*(rank+1),12], 'Cell' : [3*rank, 3*(rank+1), 6]}, dist_zone)
    quad = I.newElements('Quad', 'QUAD', quad_ec, [1,6], parent=dist_zone)
    IE.newDistribution({'Element' : [3*rank,3*(rank+1),6]}, quad)
    bar  = I.newElements('Bar', 'BAR', bar_ec, [7,11], parent=dist_zone)
    IE.newDistribution({'Element' : [0,5*(1-rank),5]}, bar)
    dist_zbc  = I.newZoneBC(dist_zone)
    if rank == 0:
      bca       = I.newBC('bcA', pointRange = [[7,9]], parent=dist_zbc)
      IE.newDistribution({'Index' : [0,2,3]}, bca)
      bcb       = I.newBC('bcB', pointList = [[10]], parent=dist_zbc)
      IE.newDistribution({'Index' : [0,1,2]}, bcb)
    elif rank == 1:
      bca       = I.newBC('bcA', pointRange = [[7,9]], parent=dist_zbc)
      IE.newDistribution({'Index' : [2,3,3]}, bca)
      bcb       = I.newBC('bcB', pointList = [[11]], parent=dist_zbc)
      IE.newDistribution({'Index' : [1,2,2]}, bcb)
    I.newGridLocation('EdgeCenter', bca)
    I.newGridLocation('EdgeCenter', bcb)

    GNG.generate_ngon_from_std_elements(dist_tree, sub_comm)

    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Quad') == quad
    assert I.getNodeFromPath(dist_tree, 'Base/Zone/Bar') == bar
    ngon  = I.getNodeFromPath(dist_tree, 'Base/Zone/NGonElements')
    nface = I.getNodeFromPath(dist_tree, 'Base/Zone/NFaceElements')

    #Apparently addition elements causes the distribution to change // dont retest here
    assert (sids.ElementRange(ngon) == [12,28]).all()
    assert (sids.ElementRange(nface) == [29,34]).all()
    assert sids.GridLocation(bca) == 'EdgeCenter'
    if rank == 0:
      assert (I.getNodeFromName(bca, 'PointList')[1] == [25,27]).all()
      assert (I.getNodeFromName(bcb, 'PointList')[1] == [14]).all()
    elif rank == 1:
      assert (I.getNodeFromName(bca, 'PointList')[1] == [28]).all()
      assert (I.getNodeFromName(bcb, 'PointList')[1] == [21]).all()

