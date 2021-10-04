from pytest_mpi_check._decorator import mark_mpi_test
from mpi4py import MPI

import Generator.PyTree   as G
import Converter.Internal as I
import numpy as np

from maia.sids     import sids
from maia.sids     import Internal_ext as IE
from maia.utils    import parse_yaml_cgns
from maia.generate import disttree_from_parttree as DFP
from maia.generate.dcube_generator import dcube_generate

def test_match_jn_from_ordinals():
  dt = """
Base CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t:
        Ordinal UserDefinedData_t [1]:
        OrdinalOpp UserDefinedData_t [2]:
        PointList IndexArray_t [[1,3]]:
      perio2 GridConnectivity_t:
        Ordinal UserDefinedData_t [2]:
        OrdinalOpp UserDefinedData_t [1]:
        PointList IndexArray_t [[2,4]]:
      match1 GridConnectivity_t:
        Ordinal UserDefinedData_t [3]:
        OrdinalOpp UserDefinedData_t [4]:
        PointList IndexArray_t [[10,100]]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t:
        Ordinal UserDefinedData_t [4]:
        OrdinalOpp UserDefinedData_t [3]:
        PointList IndexArray_t [[-100,-10]]:
  """
  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  DFP.match_jn_from_ordinals(dist_tree)
  expected_names  = ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA']
  expected_pl_opp = [[2,4], [1,3], [-100,-10], [10,100]]
  for i, jn in enumerate(I.getNodesFromType(dist_tree, 'GridConnectivity_t')):
    assert I.getValue(jn) == expected_names[i]
    assert (I.getNodeFromName1(jn, 'PointListDonor')[1] == expected_pl_opp[i]).all()

@mark_mpi_test(3)
def test_disttree_from_parttree(sub_comm):
  # > PartTree creation (cumbersome because of old ngon norm)
  # Value test is already performed in subfunction tests
  part_tree = I.newCGNSTree()
  if sub_comm.Get_rank() < 2:
    part_base = I.newCGNSBase(parent=part_tree)
    distri_ud = IE.newGlobalNumbering()
    if sub_comm.Get_rank() == 0:
      # part_zone = G.cartNGon((0,0,0), (.5,.5,.5), (3,3,3))
      part_zone = I.getZones(dcube_generate(3, 1, [0., 0., 0.], MPI.COMM_SELF))[0]
      I._rmNodesByName(part_zone, 'ZoneBC')
      I._rmNodesByName(part_zone, ':CGNS#Distribution')

      vtx_gnum = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,41,42,43]
      cell_gnum = [1,2,5,6,9,10,13,14]
      ngon_gnum = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,25,26,29,30,33,34,37,38,41,42,45,46,49,50,53,54,57,58,61,62,65,66]
      zbc = I.newZoneBC(parent=part_zone)
      bc = I.newBC(btype='BCWall', pointList=[[1,4,2,3]], parent=zbc)
      I.newGridLocation('FaceCenter', bc)
      IE.newGlobalNumbering({'Index' : [1,2,3,4]}, parent=bc)
    else:
      # part_zone = G.cartNGon((1,0,0), (.5,.5,.5), (3,3,3))
      part_zone = I.getZones(dcube_generate(3, 1, [1., 0., 0.], MPI.COMM_SELF))[0]
      I._rmNodesByName(part_zone, 'ZoneBC')
      I._rmNodesByName(part_zone, ':CGNS#Distribution')
      vtx_gnum = [3,4,5, 8,9,10,13,14,15,18,19,20,23,24,25,28,29,30,33,34,35,38,39,40,43,44,45]
      cell_gnum = [3,4,7,8,11,12,15,16]
      ngon_gnum = [3,4,5,8,9,10,13,14,15,18,19,20,23,24,27,28,31,32,35,36,39,40,43,44,47,48,51,52,55,56,59,60,63,64,67,68]

    ngon = I.getNodeFromPath(part_zone, 'NGonElements')
    IE.newGlobalNumbering({'Element' : ngon_gnum}, parent=ngon)

    #Add nface (to modify when maia become cgns compliant)
    nface_ec = np.empty(8*6, dtype=I.getNodeFromName(ngon, 'ElementConnectivity')[1].dtype)
    nface_count = np.zeros(8, int)
    for iface, (lCell, rCell) in enumerate(I.getNodeFromName1(ngon, 'ParentElements')[1]):
      nface_ec[6*(lCell-1) + nface_count[lCell-1]] = iface + 1
      nface_count[lCell-1] += 1
      if rCell > 0:
        nface_ec[6*(rCell-1) + nface_count[rCell-1]] = iface + 1
        nface_count[rCell-1] += 1
    nface = I.newElements('NFaceElements', 'NFACE', nface_ec, [36+1, 36+8], parent=part_zone)
    I.newDataArray('ElementStartOffset', np.arange(0, 6*(8+1), 6), nface)
    IE.newGlobalNumbering({'Element' : cell_gnum}, parent=nface)

    I.newDataArray('Vertex', vtx_gnum,  parent=distri_ud)
    I.newDataArray('Cell',   cell_gnum, parent=distri_ud)

    part_zone[0] = "Zone.P{0}.N0".format(sub_comm.Get_rank())
    I._addChild(part_base, part_zone)
    I._addChild(part_zone, distri_ud)

  dist_tree = DFP.disttree_from_parttree(part_tree, sub_comm)

  dist_zone = I.getNodeFromName(dist_tree, 'Zone')
  assert (dist_zone[1] == [[45,16,0]]).all()
  assert (I.getNodeFromPath(dist_zone, 'NGonElements/ElementRange')[1] == [1,68]).all()
  assert (I.getNodeFromPath(dist_zone, 'NFaceElements/ElementRange')[1] == [69,84]).all()
  assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZoneBC/BC')) == "BCWall")
