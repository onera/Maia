import pytest
import pytest_parallel
import shutil

import maia
import maia.pytree as PT
import maia.algo.dist.adaptation_utils as adapt_utils
from   maia.pytree.yaml import parse_yaml_cgns
from   maia.utils import par_utils

from maia import npy_pdm_gnum_dtype as pdm_dtype

import numpy as np

def gen_dist_zone(comm):
  if comm.rank==0:
    zone_n = PT.new_Zone('zone', type='Unstructured', size=[[9,5,0]])
    cx = np.array([1.,3.,5.,7.,9.])
    cy = np.array([0.,0.,0.,0.,0.])
    cz = np.array([0.,0.,0.,0.,0.])
    grid_coord_n = PT.new_GridCoordinates(fields={'CoordinateX':cx,'CoordinateY':cy,'CoordinateZ':cz}, parent=zone_n)
    PT.maia.newDistribution({'Vertex':[0,5,9]}, parent=zone_n)
    fields = {'cX':cx,'cY':cy,'cZ':cz}
    flowsol_n = PT.new_FlowSolution('FSolution', loc='Vertex', fields=fields, parent=zone_n)
    # PT.maia.newDistribution({'Index':[0,5,9]}, parent=flowsol_n)

  elif comm.rank==1:
    zone_n = PT.new_Zone('zone', type='Unstructured', size=[[9,5,0]])
    cx = np.array([2.,4.,6.,8.])
    cy = np.array([0.,0.,0.,0.])
    cz = np.array([0.,0.,0.,0.])
    grid_coord_n = PT.new_GridCoordinates(fields={'CoordinateX':cx,'CoordinateY':cy,'CoordinateZ':cz}, parent=zone_n)
    PT.maia.newDistribution({'Vertex':[5,9,9]}, parent=zone_n)
    fields = {'cX':cx,'cY':cy,'cZ':cz}
    flowsol_n = PT.new_FlowSolution('FSolution', loc='Vertex', fields=fields, parent=zone_n)
    # PT.maia.newDistribution({'Index':[5,9,9]}, parent=flowsol_n)

  return zone_n

@pytest_parallel.mark.parallel(2)
def test_duplicate_specified_vtx(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
    expected_cx = np.array([1., 3., 5., 7., 9., 2., 4.])
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)
    expected_cx = np.array([6., 8., 4., 8., 3., 7., 6.])

  adapt_utils.duplicate_specified_vtx(zone_n, vtx_pl, comm)

  assert np.allclose(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), expected_cx)
  assert np.allclose(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), expected_cx)

  new_distri = PT.maia.getDistribution(zone_n, 'Vertex')[1]
  assert PT.Zone.n_vtx(zone_n)==14
  assert (maia.utils.par_utils.partial_to_full_distribution(new_distri, comm) == [0,7,14]).all()


@pytest_parallel.mark.parallel(2)
def test_remove_specified_vtx(comm):
  zone_n = gen_dist_zone(comm)

  if comm.Get_rank()==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
    expected_cx = np.array([1., 5., 9.])
  elif comm.Get_rank()==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)
    expected_cx = np.array([2.])

  adapt_utils.remove_specified_vtx(zone_n, vtx_pl, comm)

  assert np.allclose(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), expected_cx)
  assert np.allclose(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), expected_cx)

  new_distri = PT.maia.getDistribution(zone_n, 'Vertex')[1]
  assert PT.Zone.n_vtx(zone_n)==4
  assert (maia.utils.par_utils.partial_to_full_distribution(new_distri, comm) == [0,3,4]).all()


@pytest_parallel.mark.parallel(2)
def test_elmt_pl_to_vtx_pl(comm):
  zone = PT.new_Zone(type='Unstructured')
  if comm.Get_rank() == 0:
    vtx_distri = np.array([0, 5, 9], pdm_dtype)
    econn      = [1,2,5,4, 2,3,6,5]
    distri     = np.array([0, 2, 4], pdm_dtype)
    elt_pl     = np.array([1,3]) # Requested elts
  elif comm.Get_rank() == 1:
    vtx_distri = np.array([5, 9, 9], pdm_dtype)
    econn      = [4,5,8,7, 5,6,9,8]
    distri     = np.array([2, 4, 4], pdm_dtype)
    elt_pl     = np.array([], pdm_dtype) # Requested elts
  elt = PT.new_Elements(type='QUAD_4', erange=[1,4], econn=econn, parent=zone)
  PT.maia.newDistribution({'Element':     distri}, elt)
  PT.maia.newDistribution({'Vertex' : vtx_distri}, zone)

  quad_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and\
                                                       PT.Element.CGNSName(n)=='QUAD_4')

  vtx_pl = adapt_utils.elmt_pl_to_vtx_pl(zone, quad_n, elt_pl, comm)
  if comm.Get_rank() == 0:
    assert (vtx_pl == [1,2,4,5]).all()
  elif comm.Get_rank() == 1:
    assert (vtx_pl == [7,8]).all()

@pytest_parallel.mark.parallel(2)
def test_tag_elmt_owning_vtx(comm):
  zone = PT.new_Zone(type='Unstructured')
  if comm.Get_rank() == 0:
    econn = [1,2,5,4, 2,3,6,5]
    distri = np.array([0, 2, 4], pdm_dtype)
    vtx_pl = np.array([1,4,5]) #Requested vtx
  elif comm.Get_rank() == 1:
    econn = [4,5,8,7, 5,6,9,8]
    distri = np.array([2, 4, 4], pdm_dtype)
    vtx_pl = np.array([2]) #Requested vtx
  elt = PT.new_Elements(type='QUAD_4', erange=[1,4], econn=econn, parent=zone)
  PT.maia.newDistribution({'Element' : distri}, elt)
  
  quad_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and\
                                                       PT.Element.CGNSName(n)=='QUAD_4')

  elt_pl = adapt_utils.tag_elmt_owning_vtx(quad_n, vtx_pl, comm, elt_full=True)
  assert (np.concatenate(comm.allgather(elt_pl)) == [1]).all()
  elt_pl = adapt_utils.tag_elmt_owning_vtx(quad_n, vtx_pl, comm, elt_full=False)
  assert (np.concatenate(comm.allgather(elt_pl)) == [1,2,3,4]).all()


@pytest_parallel.mark.parallel(2)
def test_convert_vtx_gcs_as_face_bcs(comm):

  rank = comm.Get_rank()
  econn     = np.array([1,2,3, 4,5,6], dtype=np.int32)  if rank==0 else np.array([7,8,9], dtype=np.int32)
  e_distri  = [0,2,3]                                   if rank==0 else [2,3,3]
  bc_pl     = np.array([[3]], dtype=np.int32)           if rank==0 else np.array([[2]], dtype=np.int32)
  bc_distri = [0,1,2]                                   if rank==0 else [1,2,2]
  gc0_pl    = np.array([[5,7,9]], dtype=np.int32)       if rank==0 else np.array([[4,6,8]], dtype=np.int32)
  gc1_pl    = np.array([[1]], dtype=np.int32)           if rank==0 else np.array([[2,3]], dtype=np.int32)

  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  zone_n = PT.new_Zone('zone', type='Unstructured', size=[[9,3,0]], parent=base)
  elt_n  = PT.new_Elements('TRI_3', type='TRI_3', erange=np.array([1,3], dtype=np.int32), econn=econn, parent=zone_n)
  PT.maia.newDistribution({'Element':e_distri}, parent=elt_n)
  zone_bc_n = PT.new_ZoneBC(parent=zone_n)
  bc_n = PT.new_BC('bc0', point_list=bc_pl, loc='FaceCenter', parent=zone_bc_n)
  PT.maia.newDistribution({'Index':bc_distri}, parent=bc_n)
  zone_gc_n = PT.new_ZoneGridConnectivity(parent=zone_n)
  gc_n = PT.new_GridConnectivity('gc0', type='Abutting1to1', point_list=gc0_pl, loc='Vertex', parent=zone_gc_n)
  PT.new_GridConnectivityProperty(periodic={'translation': [1.0,0]}, parent=gc_n)
  gc_n = PT.new_GridConnectivity('gc1', type='Abutting1to1', point_list=gc1_pl, loc='Vertex', parent=zone_gc_n)
  PT.new_GridConnectivityProperty(periodic={'translation': [-1.0,0]}, parent=gc_n)

  adapt_utils.convert_vtx_gcs_as_face_bcs(tree, comm)

  # > All GCs are converted, empty GCs shouldn't be transformed
  gc0 = PT.get_node_from_name_and_label(zone_n, 'gc0', 'BC_t')
  gc1 = PT.get_node_from_name_and_label(zone_n, 'gc1', 'BC_t')
  assert gc0 is None
  assert gc1 is not None
  if comm.rank==0:
    assert np.array_equal(PT.get_child_from_name(gc1, 'PointList')[1], np.array([[1]], dtype=np.int32))
  if comm.rank==1:
    assert np.array_equal(PT.get_child_from_name(gc1, 'PointList')[1], np.array([[]], dtype=np.int32))


def test_apply_offset_to_elts():
  yt = f"""
    Zone Zone_t:
      BAR Elements_t I4 [3, 0]:
        ElementRange IndexRange_t I4 [23, 30]:
      TRI_1 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [1, 5]:
      TETRA Elements_t I4 [10, 0]:
        ElementRange IndexRange_t I4 [6, 8]:
      TRI_2 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [11, 22]:
      TRI_3 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [31, 31]:
      ZoneBC ZoneBC_t:
        BC1 BC_t 'Null':
          GridLocation GridLocation_t 'CellCenter':
          PointList IndexArray_t I4 [[6,7,8]]:
        BC2 BC_t 'Null':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I4 [[1,2,11,22]]:
        BC3 BC_t 'Null':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I4 [[1,3,5]]:
        BC4 BC_t 'Null':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I4 [[23, 24, 25]]:
    """
  zone = parse_yaml_cgns.to_node(yt)
  elt_n = PT.get_child_from_name(zone, 'TETRA')
  min_range = PT.Element.Range(elt_n)[1]
  maia.algo.dist.adaptation_utils.apply_offset_to_elts(zone, -2, min_range)

  expected_elt_er = { 'TRI_1': np.array([1,5]),
                      'TETRA': np.array([6,8]),
                      'TRI_2': np.array([9,20]),
                      'BAR'  : np.array([21,28]),
                      'TRI_3': np.array([29,29]),
                      }
  for elt_name, elt_er in expected_elt_er.items():
    elt_n = PT.get_child_from_name(zone, elt_name)
    assert np.array_equal(PT.Element.Range(elt_n), elt_er)

  expected_bc_pl = {'BC1': np.array([[6,7,8]]),
                    'BC2': np.array([[1,2,9,20]]),
                    'BC3': np.array([[1,3,5]]),
                    'BC4': np.array([[21,22,23]]),
                     }
  for bc_name, bc_pl in expected_bc_pl.items():
    bc_n = PT.get_node_from_name_and_label(zone, bc_name, 'BC_t')
    assert np.array_equal(PT.Subset.getPatch(bc_n)[1], bc_pl)


@pytest_parallel.mark.parallel(2)
def test_is_elt_included(comm):
  tree = maia.factory.generate_dist_block(3, 'TETRA_4', comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')

  # This is the selected TRI (57,58,59,60)   : 1 10 4    13 4 10    7 4 16    13 16 4
  # This is the selected TETRA (2,3)         : 11 10 14 2    13 14 10 4
  # Decomposed faces of TRI are :  11 10 14   11 14 2   11 10 2  10 14 2
  #                                13 14 10   13 14 4   13 10 4  14 10 4
  # Only face TRI 58 appreas in tetra faces             ^ Here
  
  if comm.Get_rank() == 0:
    tri_pl = np.array([57,58])
    tetra_pl = np.array([2])
  else:
    tri_pl = np.array([59, 60])
    tetra_pl = np.array([3])

  tri_elt   = PT.get_node_from_predicate(zone, lambda n : PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3')
  tetra_elt = PT.get_node_from_predicate(zone, lambda n : PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4')
  out = adapt_utils.find_shared_faces(tri_elt, tri_pl, tetra_elt, tetra_pl, comm)

  if comm.Get_rank() == 0:
    assert (out == [58]).all()
  else:
    assert (out == []).all()

@pytest_parallel.mark.parallel([1,2,3])
def test_add_undefined_faces(comm):
  dist_tree = maia.factory.dcube_generator.dcube_nodal_generate(3, 1., [0.,0.,0.], 'TETRA_4', comm, get_ridges=True)
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  tet_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4')
  tri_n = PT.get_child_from_predicate(zone, lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3')

  zmax_n  = PT.get_node_from_path(zone, 'ZoneBC/Zmax')
  zmax_pl = PT.Subset.getPatch(zmax_n)[1][0]
  vtx_pl  = adapt_utils.elmt_pl_to_vtx_pl(zone, tri_n, zmax_pl, comm)
  cell_pl = adapt_utils.tag_elmt_owning_vtx(tet_n, vtx_pl, comm, elt_full=False)

  adapt_utils.add_undefined_faces(zone, tet_n, cell_pl, tri_n, comm)
  
  assert PT.Zone.n_vtx(zone)==27
  assert PT.Zone.n_cell(zone)==40
  assert np.array_equal(PT.Element.Range(tet_n),np.array([ 1,40], dtype=np.int32))
  assert PT.maia.getDistribution(tet_n, 'Element')[1][2]==40
  assert np.array_equal(PT.Element.Range(tri_n),np.array([41,96], dtype=np.int32))
  assert PT.maia.getDistribution(tri_n, 'Element')[1][2]==56


@pytest_parallel.mark.parallel([1,3])
def test_find_matching_bcs(comm):

  def distributed(array):
    distri = par_utils.uniform_distribution(array.size, comm)
    return array[distri[0]:distri[1]]

  distri_bar = par_utils.uniform_distribution(10, comm)
  
  bar_ec  = np.array([1,2, 2,3, 4,1, 7,4,  3,6, 6,9, 101,102, 102,103, 104,101, 107,104])[distri_bar[0]*2:distri_bar[1]*2]
  bc1_pl  = distributed(np.array([1,2,4]))
  bc2_pl  = distributed(np.array([8,7,10]))
  bc3_pl  = distributed(np.array([1,4]))  # Does not fully match with 8,7,10
  bc3_pl  = distributed(np.array([5,6])) # Does not match at all
  gc1_vtx = distributed(np.array([1,2,3,4,5,6,7,8,9], np.int32))
  gc2_vtx = distributed(np.array([101,102,103,104,105,106,107,108,109], np.int32))
  
  zone_n = PT.new_Zone('zone', type='Unstructured', size=[[120,3,0]])
  PT.maia.newDistribution({'Vertex':par_utils.uniform_distribution(120, comm)}, parent=zone_n)
  bar_n  = PT.new_Elements('BAR_2', type='BAR_2', erange=[1,10], econn=bar_ec, parent=zone_n)
  PT.maia.newDistribution({'Element':distri_bar}, parent=bar_n)
  zbc_n = PT.new_ZoneBC(parent=zone_n)
  PT.new_BC('BC1', point_list=bc1_pl.reshape((1,-1),order='F'), loc='EdgeCenter', parent=zbc_n)
  PT.new_BC('BC2', point_list=bc2_pl.reshape((1,-1),order='F'), loc='EdgeCenter', parent=zbc_n)
  PT.new_BC('BC3', point_list=bc3_pl.reshape((1,-1),order='F'), loc='EdgeCenter', parent=zbc_n)
    
  src_pl = distributed(np.array([1,2,3,4]))
  tgt_pl = distributed(np.array([7,8,9,10]))

  matching_bcs = adapt_utils.find_matching_bcs(zone_n, bar_n,
                                               src_pl, tgt_pl,
                                               [gc1_vtx,gc2_vtx],
                                               comm)
  assert matching_bcs==[['BC2','BC1']]


@pytest.mark.parametrize('partial',[False, True])
@pytest_parallel.mark.parallel([1,2,3])
def test_update_elt_vtx_numbering(partial, comm):
  distri_bar = par_utils.uniform_distribution(9, comm)
  distri_vtx = par_utils.uniform_distribution(12, comm)
  distri_num = par_utils.uniform_distribution(12, comm)
  
  bar_ec  = np.array([5,6, 7,8, 9,10, 1,2, 9,10, 2,4, 4,3, 8,10, 11,12])[distri_bar[0]*2:distri_bar[1]*2]
  
  zone_n = PT.new_Zone('zone', type='Unstructured', size=[[12,3,0]])
  bar_n  = PT.new_Elements('BAR_2', type='BAR_2', erange=np.array([1,6], dtype=np.int32), econn=bar_ec, parent=zone_n)
  PT.maia.newDistribution({'Vertex' :distri_vtx}, parent=zone_n)
  PT.maia.newDistribution({'Element':distri_bar}, parent=bar_n)

  old_to_new = np.array([1,6,3,5,5,6,7,8,9,8,9,12])[distri_num[0]:distri_num[1]]
  
  if partial:
    distri_pl = par_utils.uniform_distribution(4, comm)
    bar_pl    = np.array([4,1,7,9])[distri_pl[0]:distri_pl[1]]
  else:
    bar_pl    = None

  adapt_utils.update_elt_vtx_numbering(zone_n, bar_n, old_to_new, comm, bar_pl)
  bar_ec = PT.get_value(PT.get_child_from_name(bar_n, 'ElementConnectivity'))
  
  if partial:
    expected_ec = np.array([5,6, 7,8, 9,10, 1,6, 9,10, 2,4, 5,3, 8,10, 9,12])[distri_bar[0]*2:distri_bar[1]*2]
  else:
    expected_ec = np.array([5,6, 7,8, 9,8, 1,6, 9,8, 6,5, 5,3, 8,8, 9,12])[distri_bar[0]*2:distri_bar[1]*2]

  assert np.array_equal(bar_ec, expected_ec)
