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
    expected_cx = np.array([1., 3., 5., 7., 9., 4., 8.])
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)
    expected_cx = np.array([2., 4., 6., 8., 3., 7., 6.])

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
    econn = [1,2,5,4, 2,3,6,5]
    distri = np.array([0, 2, 4], pdm_dtype)
    elt_pl = np.array([1,3]) # Requested elts
  elif comm.Get_rank() == 1:
    econn = [4,5,8,7, 5,6,9,8]
    distri = np.array([2, 4, 4], pdm_dtype)
    elt_pl = np.array([], pdm_dtype) # Requested elts
  elt = PT.new_Elements(type='QUAD_4', erange=[1,4], econn=econn, parent=zone)
  PT.maia.newDistribution({'Element' : distri}, elt)

  vtx_pl = adapt_utils.elmt_pl_to_vtx_pl(zone, elt_pl, 'QUAD_4', comm)
  if comm.Get_rank() == 0:
    assert (vtx_pl == [1,2,4,5,7,8]).all()
  elif comm.Get_rank() == 1:
    assert vtx_pl.size == 0

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

  elt_pl = adapt_utils.tag_elmt_owning_vtx(zone, vtx_pl, 'QUAD_4', comm, elt_full=True)
  assert (np.concatenate(comm.allgather(elt_pl)) == [1]).all()
  elt_pl = adapt_utils.tag_elmt_owning_vtx(zone, vtx_pl, 'QUAD_4', comm, elt_full=False)
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
