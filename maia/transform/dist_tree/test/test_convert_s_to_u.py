import pytest
import mpi4py.MPI as MPI
import numpy as np
from maia.utils import parse_yaml_cgns
from maia.transform.dist_tree import convert_s_to_u
import Converter.Internal as I

# class Test_compare_pointrange():
#   def test_ok(self):
#     jn1 = I.newGridConnectivity1to1(pointRange=[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
#     jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange=[[7,1],[9,9],[5,1]])
#     assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == True)
#   def test_ko(self):
#     jn1 = I.newGridConnectivity1to1(pointRange=[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
#     jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange=[[1,7],[9,9],[1,5]])
#     assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == False)
# class Test_MPI():
#   @pytest.mark.mpi(min_size=3)
#   @pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
#   def test_mpi(self,sub_comm):
#     if(sub_comm == MPI.COMM_NULL):
#       return
#     nRank = sub_comm.Get_size()
#     iRank = sub_comm.Get_rank()

###############################################################################
def test_convert_ijk_to_index():
  assert convert_s_to_u.convert_ijk_to_index(1,1,1,3,3,3) ==   1
  assert convert_s_to_u.convert_ijk_to_index(1,2,3,7,5,3) ==  78
  assert convert_s_to_u.convert_ijk_to_index(7,5,3,7,5,3) == 105
###############################################################################

###############################################################################
def test_convert_ijk_to_faceiIndex():
  assert convert_s_to_u.convert_ijk_to_faceiIndex(1,1,1,[7,5,3],[8,6,4]) ==   1
  assert convert_s_to_u.convert_ijk_to_faceiIndex(3,2,1,[7,5,3],[8,6,4]) ==  11
  assert convert_s_to_u.convert_ijk_to_faceiIndex(7,5,3,[7,5,3],[8,6,4]) == 119
  assert convert_s_to_u.convert_ijk_to_faceiIndex(8,5,3,[7,5,3],[8,6,4]) == 120
###############################################################################

###############################################################################
def test_convert_ijk_to_facejIndex():
  assert convert_s_to_u.convert_ijk_to_facejIndex(1,1,1,[7,5,3],[8,6,4]) == 121
  assert convert_s_to_u.convert_ijk_to_facejIndex(3,2,1,[7,5,3],[8,6,4]) == 130
  assert convert_s_to_u.convert_ijk_to_facejIndex(7,5,3,[7,5,3],[8,6,4]) == 239
  assert convert_s_to_u.convert_ijk_to_facejIndex(7,6,3,[7,5,3],[8,6,4]) == 246
###############################################################################

###############################################################################
def test_convert_ijk_to_facekIndex():
  assert convert_s_to_u.convert_ijk_to_facekIndex(1,1,1,[7,5,3],[8,6,4]) == 247
  assert convert_s_to_u.convert_ijk_to_facekIndex(3,2,1,[7,5,3],[8,6,4]) == 256
  assert convert_s_to_u.convert_ijk_to_facekIndex(7,5,3,[7,5,3],[8,6,4]) == 351
  assert convert_s_to_u.convert_ijk_to_facekIndex(7,5,4,[7,5,3],[8,6,4]) == 386
###############################################################################
  
###############################################################################
def test_convert_ijk_to_faceIndices():
  assert convert_s_to_u.convert_ijk_to_faceIndices(1,1,1,[7,5,3],[8,6,4]) == (  1,121,247)
  assert convert_s_to_u.convert_ijk_to_faceIndices(3,2,1,[7,5,3],[8,6,4]) == ( 11,130,256)
  assert convert_s_to_u.convert_ijk_to_faceIndices(7,5,3,[7,5,3],[8,6,4]) == (119,239,351)
###############################################################################
  
###############################################################################
def test_compute_fi_from_ijk():
  assert convert_s_to_u.compute_fi_from_ijk(5,4,3)    == ((5,4,3),(5,5,3),(5,5,4),(5,4,4),(4,4,3),(5,4,3))
###############################################################################
  
###############################################################################
def test_compute_fi_from_imaxjk():
  assert convert_s_to_u.compute_fi_from_imaxjk(6,4,3) == ((6,4,3),(6,5,3),(6,5,4),(6,4,4),(5,4,3),0)
###############################################################################
  
###############################################################################
def test_compute_fi_from_iminjk():
  assert convert_s_to_u.compute_fi_from_iminjk(1,4,3) == ((1,4,3),(1,4,4),(1,5,4),(1,5,3),(1,4,3),0)
###############################################################################
  
###############################################################################
def test_compute_fj_from_ijk():
  assert convert_s_to_u.compute_fj_from_ijk(5,4,3)    == ((5,4,3),(5,4,4),(6,4,4),(6,4,3),(5,3,3),(5,4,3))
###############################################################################
  
###############################################################################
def test_compute_fj_from_ijmaxk():
  assert convert_s_to_u.compute_fj_from_ijmaxk(5,5,3) == ((5,5,3),(5,5,4),(6,5,4),(6,5,3),(5,4,3),0)
###############################################################################
  
###############################################################################
def test_compute_fj_from_ijmink():
  assert convert_s_to_u.compute_fj_from_ijmink(5,1,3) == ((5,1,3),(6,1,3),(6,1,4),(5,1,4),(5,1,3),0)
###############################################################################
  
###############################################################################
def test_compute_fk_from_ijk():
  assert convert_s_to_u.compute_fk_from_ijk(5,4,3)    == ((5,4,3),(6,4,3),(6,5,3),(5,5,3),(5,4,2),(5,4,3))
###############################################################################
  
###############################################################################
def test_compute_fk_from_ijkmax():
  assert convert_s_to_u.compute_fk_from_ijkmax(5,4,4) == ((5,4,4),(6,4,4),(6,5,4),(5,5,4),(5,4,3),0)
###############################################################################
  
###############################################################################
def test_compute_fk_from_ijkmin():
  assert convert_s_to_u.compute_fk_from_ijkmin(5,4,1) == ((5,4,1),(5,5,1),(6,5,1),(6,4,1),(5,4,1),0)
###############################################################################
  
###############################################################################
class Test_fill_faceNgon_leftCell_rightCell():
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fi_ijk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 0
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,4,3),(5,5,3),(5,5,4),(5,4,4),
                                                    (4,4,3),(5,4,3),
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 83, 89,119,113]).all()
        assert faceLeftCell[f]        == 59
        assert faceRightCell[f]       == 60
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fi_imaxjk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 1
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (6,4,3),(6,5,3),(6,5,4),(6,4,4),
                                                    (5,4,3),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 84, 90,120,114]).all()
        assert faceLeftCell[f]        == 60
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fi_iminjk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 2
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (1,4,3),(1,4,4),(1,5,4),(1,5,3),
                                                    (1,4,3),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 79,109,115, 85]).all()
        assert faceLeftCell[f]        == 56
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fj_ijk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 3
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,4,3),(5,4,4),(6,4,4),(6,4,3),
                                                    (5,3,3),(5,4,3),
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 83,113,114, 84]).all()
        assert faceLeftCell[f]        == 55
        assert faceRightCell[f]       == 60
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fj_ijmaxk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 4
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,5,3),(5,5,4),(6,5,4),(6,5,3),
                                                    (5,4,3),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,
                                                    faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 89,119,120, 90]).all()
        assert faceLeftCell[f]        == 60
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fj_ijmink(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 5
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,1,3),(6,1,3),(6,1,4),(5,1,4),
                                                    (5,1,3),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 65, 66, 96, 95]).all()
        assert faceLeftCell[f]        == 45
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fk_ijk(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 6
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,4,3),(6,4,3),(6,5,3),(5,5,3),
                                                    (5,4,2),(5,4,3),
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 83, 84, 90, 89]).all()
        assert faceLeftCell[f]        == 40
        assert faceRightCell[f]       == 60
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fk_ijkmax(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 7
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,4,4),(6,4,4),(6,5,4),(5,5,4),
                                                    (5,4,3),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [113,114,120,119]).all()
        assert faceLeftCell[f]        == 60
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
# --------------------------------------------------------------------------- #
  def test_fill_faceNgon_leftCell_rightCell_fk_ijkmin(self):
    nbFacesAllSlabsPerZone = 9
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    counter = 8
    convert_s_to_u.fill_faceNgon_leftCell_rightCell(counter,
                                                    (5,4,1),(5,5,1),(6,5,1),(6,4,1),
                                                    (5,4,1),0,
                                                    [6,5,4],[5,4,3],
                                                    faceNgon,faceLeftCell,faceRightCell)
    for f in range(nbFacesAllSlabsPerZone):
      if f == counter:
        assert (faceNgon[4*f:4*(f+1)] == [ 23, 29, 30, 24]).all()
        assert faceLeftCell[f]        == 20
        assert faceRightCell[f]       ==  0
      else:
        assert (faceNgon[4*f:4*(f+1)] == [ -1, -1, -1, -1]).all()
        assert faceLeftCell[f]        == -1
        assert faceRightCell[f]       == -1
###############################################################################
  
###############################################################################
class Test_compute_nbFacesAllSlabsPerZone():
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_monoslab_imax_jmax_kmax(self):
    slabListVtx = [[[0, 3], [0, 3], [2, 3]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx,nVtx) == 4
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_monoslab_imin_jmin_kmin(self):
    slabListVtx = [[[0, 3], [0, 3], [0, 1]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx,nVtx) == 16
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_monoslab_random(self):
    slabListVtx = [[[1, 2], [0, 1], [1, 2]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx,nVtx) == 3
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs1(self):
    slabListVtx = [[[1, 3], [1, 2], [2, 3]], [[0, 3], [2, 3], [2, 3]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[0]],nVtx) == 1
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[1]],nVtx) == 0
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx     ,nVtx) == 1+0
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs2(self):
    slabListVtx = [[[0, 3], [1, 2], [1, 2]], [[0, 2], [2, 3], [1, 2]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[0]],nVtx) == 7
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[1]],nVtx) == 2
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx     ,nVtx) == 7+2
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs3(self):
    slabListVtx = [[[2, 3], [2, 3], [1, 2]], [[0, 3], [0, 1], [2, 3]], [[0, 1], [1, 2], [2, 3]]]
    nVtx        = [3, 3, 3]
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[0]],nVtx) == 0
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[1]],nVtx) == 2
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone([slabListVtx[2]],nVtx) == 1
    assert convert_s_to_u.compute_nbFacesAllSlabsPerZone(slabListVtx     ,nVtx) == 0+2+1
###############################################################################
  
###############################################################################
class Test_compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces():
# --------------------------------------------------------------------------- #
  def test_compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces_monoslab_imax_jmax_kmax(self):
    slabListVtx = [[[0, 3], [0, 3], [2, 3]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 4
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)    
    assert (faceNumber[:]    == [33,35,34,36]                                    ).all()
    assert (faceNgon[:]      == [19,20,23,22,22,23,26,25,20,21,24,23,23,24,27,26]).all()
    assert (faceLeftCell[:]  == [ 5, 7, 6, 8]                                    ).all()
    assert (faceRightCell[:] == [ 0, 0, 0, 0]                                    ).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces_monoslab_imin_jmin_kmin(self):
    slabListVtx = [[[0, 3], [0, 3], [0, 1]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 16
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)
    assert (faceNumber[:]    == [ 1, 4,13,14,25,27,26,28,15, 2,16, 5, 3, 6,17,18]).all()
    assert (faceNgon[:]      == [ 1,10,13, 4, 4,13,16, 7, 1, 2,11,10, 2, 3,12,11,
                                  1, 4, 5, 2, 4, 7, 8, 5, 2, 5, 6, 3, 5, 8, 9, 6,
                                  4,13,14, 5, 2, 5,14,11, 5,14,15, 6, 5, 8,17,14,
                                  3, 6,15,12, 6, 9,18,15, 7,16,17, 8, 8,17,18, 9]).all()
    assert (faceLeftCell[:]  == [ 1, 3, 1, 2, 1, 3, 2, 4, 1, 1, 2, 3, 2, 4, 3, 4]).all()
    assert (faceRightCell[:] == [ 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 4, 4, 0, 0, 0, 0]).all()
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_monoslab_random(self):
    slabListVtx = [[[1, 2], [0, 1], [1, 2]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 3
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)
    assert (faceNumber[:]    == [20, 8,30]                           ).all()
    assert (faceNgon[:]      == [11,12,21,20,11,14,23,20,11,12,15,14]).all()
    assert (faceLeftCell[:]  == [ 6, 5, 2]                           ).all()
    assert (faceRightCell[:] == [ 0, 6, 6]                           ).all()
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs1(self):
    slabListVtx = [[[1, 3], [1, 2], [2, 3]], [[0, 3], [2, 3], [2, 3]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 1
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)
    assert (faceNumber[:]    == [36]         ).all()
    assert (faceNgon[:]      == [23,24,27,26]).all()
    assert (faceLeftCell[:]  == [ 8]         ).all()
    assert (faceRightCell[:] == [ 0]         ).all()
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs2(self):
    slabListVtx = [[[0, 3], [1, 2], [1, 2]], [[0, 2], [2, 3], [1, 2]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 9
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)
    assert (faceNumber[:]    == [10,11,22,32,21,31,12,23,24]         ).all()
    assert (faceNgon[:]      == [13,22,25,16,14,17,26,23,14,23,24,15,
                                 14,15,18,17,13,22,23,14,13,14,17,16,
                                 15,18,27,24,16,25,26,17,17,26,27,18]).all()
    assert (faceLeftCell[:]  == [ 7, 7, 6, 4, 5, 3, 8, 7, 8]         ).all()
    assert (faceRightCell[:] == [ 0, 8, 8, 8, 7, 7, 0, 0, 0]         ).all()
# --------------------------------------------------------------------------- #
  def test_compute_nbFacesAllSlabsPerZone_multislabs3(self):
    slabListVtx = [[[2, 3], [2, 3], [1, 2]], [[0, 3], [0, 1], [2, 3]], [[0, 1], [1, 2], [2, 3]]]
    nVtx        = [3, 3, 3]
    nCell       = [2, 2, 2]
    nbFacesAllSlabsPerZone = 3
    faceNumber             = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon               = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell           = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell          = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    convert_s_to_u.compute_faceNumber_faceNgon_leftCell_rightCell_forAllFaces(slabListVtx,nVtx,nCell,
                                                                              faceNumber,faceNgon,
                                                                              faceLeftCell,faceRightCell)
    assert (faceNumber[:]    == [33,34,35]                           ).all()
    assert (faceNgon[:]      == [19,20,23,22,20,21,24,23,22,23,26,25]).all()
    assert (faceLeftCell[:]  == [ 5, 6, 7]                           ).all()
    assert (faceRightCell[:] == [ 0, 0, 0]                           ).all()
###############################################################################

###############################################################################
class Test_compute_faceList_from_vertexRange():
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_i_1rank(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,1,nCell,nVtx)
    expected_pointList = [[3,6,9,12]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,4))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_i_3ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,3,nCell,nVtx)
    pointList1 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,1,3,nCell,nVtx)
    pointList2 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,2,3,nCell,nVtx)
    expected_pointList = [[3,6,9,12]]
    assert (pointList0 == [[3,6]]).all()
    assert (pointList1 == [[  9]]).all()
    assert (pointList2 == [[ 12]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_i_5ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,5,nCell,nVtx)
    pointList1 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,1,5,nCell,nVtx)
    pointList2 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,2,5,nCell,nVtx)
    pointList3 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,3,5,nCell,nVtx)
    pointList4 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,4,5,nCell,nVtx)
    expected_pointList = [[3,6,9,12]]
    assert (pointList0 == [[ 3]]).all()
    assert (pointList1 == [[ 6]]).all()
    assert (pointList2 == [[ 9]]).all()
    assert (pointList3 == [[12]]).all()
    assert (pointList4 == [[  ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    assert (pointList3.shape == (1,1))
    assert (pointList4.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2,pointList3,pointList4),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_j_1rank(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,1,nCell,nVtx)
    expected_pointList = [[13,19]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,2))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_j_3ranks(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,3,nCell,nVtx)
    pointList1 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,1,3,nCell,nVtx)
    pointList2 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,2,3,nCell,nVtx)
    expected_pointList = [[13,19]]
    assert (pointList0 == [[13]]).all()
    assert (pointList1 == [[19]]).all()
    assert (pointList2 == [[  ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,2))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_k_1rank(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,1,nCell,nVtx)
    expected_pointList = [[25]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_k_3ranks(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,3,nCell,nVtx)
    pointList1 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,1,3,nCell,nVtx)
    pointList2 = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,2,3,nCell,nVtx)
    expected_pointList = [[25]]
    assert (pointList0 == [[25]]).all()
    assert (pointList1 == [[  ]]).all()
    assert (pointList2 == [[  ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,0))
    assert (pointList2.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,1))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_vertexRange_error(self):
    pointRange = np.reshape([1,2,1,2,1,2],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    result = False
    try:
      pointList  = convert_s_to_u.compute_faceList_from_vertexRange(pointRange,0,1,nCell,nVtx)
    except ValueError:
      result = True
    assert result
###############################################################################

###############################################################################
class Test_compute_vertexList_from_vertexRange():
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_i_1rank(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    pointList  = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,1,nVtx)
    expected_pointList = [[3,6,9,12,15,18,21,24,27]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,9))
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_i_3ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    pointList0 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,3,nVtx)
    pointList1 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,1,3,nVtx)
    pointList2 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,2,3,nVtx)
    expected_pointList = [[3,6,9,12,15,18,21,24,27]]
    assert (pointList0 == [[ 3, 6, 9]]).all()
    assert (pointList1 == [[12,15,18]]).all()
    assert (pointList2 == [[21,24,27]]).all()
    assert (pointList0.shape == (1,3))
    assert (pointList1.shape == (1,3))
    assert (pointList2.shape == (1,3))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,9))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_i_5ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nVtx       = [3, 3, 3]
    pointList0 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,5,nVtx)
    pointList1 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,1,5,nVtx)
    pointList2 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,2,5,nVtx)
    pointList3 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,3,5,nVtx)
    pointList4 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,4,5,nVtx)
    expected_pointList = [[3,6,9,12,15,18,21,24,27]]
    assert (pointList0 == [[ 3, 6]]).all()
    assert (pointList1 == [[ 9,12]]).all()
    assert (pointList2 == [[15,18]]).all()
    assert (pointList3 == [[21,24]]).all()
    assert (pointList4 == [[   27]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,2))
    assert (pointList2.shape == (1,2))
    assert (pointList3.shape == (1,2))
    assert (pointList4.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2,pointList3,pointList4),axis=None)
    pointList = np.reshape(pointList,(1,9))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_j_1rank(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nVtx       = [3, 3, 3]
    pointList  = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,1,nVtx)
    expected_pointList = [[1,2,10,11,19,20]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,6))
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_j_3ranks(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nVtx       = [3, 3, 3]
    pointList0 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,3,nVtx)
    pointList1 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,1,3,nVtx)
    pointList2 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,2,3,nVtx)
    expected_pointList = [[1,2,10,11,19,20]]
    assert (pointList0 == [[ 1, 2]]).all()
    assert (pointList1 == [[10,11]]).all()
    assert (pointList2 == [[19,20]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,2))
    assert (pointList2.shape == (1,2))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,6))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_k_1rank(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nVtx       = [3, 3, 3]
    pointList  = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,1,nVtx)
    expected_pointList = [[1,2,4,5]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,4))
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_k_3ranks(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nVtx       = [3, 3, 3]
    pointList0 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,3,nVtx)
    pointList1 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,1,3,nVtx)
    pointList2 = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,2,3,nVtx)
    expected_pointList = [[1,2,4,5]]
    assert (pointList0 == [[1,2]]).all()
    assert (pointList1 == [[  4]]).all()
    assert (pointList2 == [[  5]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_vertexList_from_vertexRange_error(self):
    pointRange = np.reshape([1,2,1,2,1,2],(3,2))
    nVtx       = [3, 3, 3]
    result = False
    try:
      pointList  = convert_s_to_u.compute_vertexList_from_vertexRange(pointRange,0,1,nVtx)
    except ValueError:
      result = True
    assert result
###############################################################################

###############################################################################
class Test_compute_cellList_from_vertexRange():
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_i_1rank(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    expected_pointList = [[2,4,6,8]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,4))
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_i_3ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,3,nCell)
    pointList1 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,1,3,nCell)
    pointList2 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,2,3,nCell)
    expected_pointList = [[2,4,6,8]]
    assert (pointList0 == [[2,4]]).all()
    assert (pointList1 == [[  6]]).all()
    assert (pointList2 == [[  8]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_i_5ranks(self):
    pointRange = np.reshape([3,3,1,3,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,5,nCell)
    pointList1 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,1,5,nCell)
    pointList2 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,2,5,nCell)
    pointList3 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,3,5,nCell)
    pointList4 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,4,5,nCell)
    expected_pointList = [[2,4,6,8]]
    assert (pointList0 == [[2]]).all()
    assert (pointList1 == [[4]]).all()
    assert (pointList2 == [[6]]).all()
    assert (pointList3 == [[8]]).all()
    assert (pointList4 == [[ ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    assert (pointList3.shape == (1,1))
    assert (pointList4.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2,pointList3,pointList4),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_j_1rank(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    expected_pointList = [[1,5]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,2))
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_j_3ranks(self):
    pointRange = np.reshape([2,3,1,1,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,3,nCell)
    pointList1 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,1,3,nCell)
    pointList2 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,2,3,nCell)
    expected_pointList = [[1,5]]
    assert (pointList0 == [[1]]).all()
    assert (pointList1 == [[5]]).all()
    assert (pointList2 == [[ ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,2))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_jmax_1rank(self):
    pointRange = np.reshape([2,3,3,3,1,3],(3,2))
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    expected_pointList = [[3,7]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,2))
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_k_1rank(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    expected_pointList = [[1]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_k_3ranks(self):
    pointRange = np.reshape([2,3,1,2,1,1],(3,2))
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,3,nCell)
    pointList1 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,1,3,nCell)
    pointList2 = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,2,3,nCell)
    expected_pointList = [[1]]
    assert (pointList0 == [[1]]).all()
    assert (pointList1 == [[ ]]).all()
    assert (pointList2 == [[ ]]).all()
    assert (pointList0.shape == (1,1))
    assert (pointList1.shape == (1,0))
    assert (pointList2.shape == (1,0))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,1))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_kmax_1rank(self):
    pointRange = np.reshape([2,3,1,2,3,3],(3,2))
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    expected_pointList = [[5]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_cellList_from_vertexRange_error(self):
    pointRange = np.reshape([1,2,1,2,1,2],(3,2))
    nCell      = [2, 2, 2]
    result = False
    try:
      pointList  = convert_s_to_u.compute_cellList_from_vertexRange(pointRange,0,1,nCell)
    except ValueError:
      result = True
    assert result
###############################################################################

###############################################################################
class Test_compute_faceList_from_faceRange():
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_i_1rank(self):
    pointRange = np.reshape([3,3,1,1,1,1],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,1,nCell,nVtx,"IFaceCenter")
    print(pointList)
    expected_pointList = [[3]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_i_3rank(self):
    pointRange = np.reshape([3,3,1,2,1,2],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,3,nCell,nVtx,"IFaceCenter")
    pointList1 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,1,3,nCell,nVtx,"IFaceCenter")
    pointList2 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,2,3,nCell,nVtx,"IFaceCenter")
    expected_pointList = [[3,6,9,12]]
    assert (pointList0 == [[3, 6]]).all()
    assert (pointList1 == [[   9]]).all()
    assert (pointList2 == [[  12]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_j_1rank(self):
    pointRange = np.reshape([3,3,1,1,1,1],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,1,nCell,nVtx,"JFaceCenter")
    expected_pointList = [[13]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_j_3rank(self):
    pointRange = np.reshape([1,2,3,3,1,2],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,3,nCell,nVtx,"JFaceCenter")
    pointList1 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,1,3,nCell,nVtx,"JFaceCenter")
    pointList2 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,2,3,nCell,nVtx,"JFaceCenter")
    expected_pointList = [[17,18,23,24]]
    assert (pointList0 == [[17,18]]).all()
    assert (pointList1 == [[   23]]).all()
    assert (pointList2 == [[   24]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_k_1rank(self):
    pointRange = np.reshape([3,3,1,1,1,1],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList  = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,1,nCell,nVtx,"KFaceCenter")
    expected_pointList = [[25]]
    assert (pointList == expected_pointList).all()
    assert (pointList.shape == (1,1))
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_k_3rank(self):
    pointRange = np.reshape([1,2,1,2,3,3],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    pointList0 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,3,nCell,nVtx,"KFaceCenter")
    pointList1 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,1,3,nCell,nVtx,"KFaceCenter")
    pointList2 = convert_s_to_u.compute_faceList_from_faceRange(pointRange,2,3,nCell,nVtx,"KFaceCenter")
    expected_pointList = [[33,34,35,36]]
    assert (pointList0 == [[33,34]]).all()
    assert (pointList1 == [[   35]]).all()
    assert (pointList2 == [[   36]]).all()
    assert (pointList0.shape == (1,2))
    assert (pointList1.shape == (1,1))
    assert (pointList2.shape == (1,1))
    pointList = np.concatenate((pointList0,pointList1,pointList2),axis=None)
    pointList = np.reshape(pointList,(1,4))
    assert (pointList == expected_pointList).all()
# --------------------------------------------------------------------------- #
  def test_compute_faceList_from_faceRange_error(self):
    pointRange = np.reshape([1,1,1,2,1,2],(3,2))
    nVtx       = [3, 3, 3]
    nCell      = [2, 2, 2]
    result = False
    try:
      pointList  = convert_s_to_u.compute_faceList_from_faceRange(pointRange,0,1,nCell,nVtx,'ZFaceCenter')
    except ValueError:
      result = True
    assert result
###############################################################################
  
###############################################################################
def test_isSameAxis():
  assert convert_s_to_u.isSameAxis( 1, 2) == 0
  assert convert_s_to_u.isSameAxis( 1, 1) == 1
  assert convert_s_to_u.isSameAxis(-2, 2) == 1
  assert convert_s_to_u.isSameAxis( 1,-1) == 1
###############################################################################
  
###############################################################################
class Test_compute_transformMatrix():
# --------------------------------------------------------------------------- #
  def test_compute_transformMatrix(self):
    transform = [1,2,3]
    attendedTransformMatrix = np.eye(3, dtype=np.int32)
    assert (convert_s_to_u.compute_transformMatrix(transform) == attendedTransformMatrix).all()
# --------------------------------------------------------------------------- #
  def test_compute_transformMatrix(self):
    transform = [-2,3,1]
    attendedTransformMatrix = np.zeros((3,3),dtype=np.int32,order='F')
    attendedTransformMatrix[0][2] =  1
    attendedTransformMatrix[1][0] = -1
    attendedTransformMatrix[2][1] =  1
    assert (convert_s_to_u.compute_transformMatrix(transform) == attendedTransformMatrix).all()
###############################################################################

    
    
    
# --------------------------------------------------------------------------- #
  # @pytest.mark.mpi(min_size=3)
  # @pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
  # def test_mpi(self,sub_comm):
  #   if(sub_comm == MPI.COMM_NULL):
  #     return
  #   nRank = sub_comm.Get_size()
  #   iRank = sub_comm.Get_rank()
    
    
    
    

###############################################################################
