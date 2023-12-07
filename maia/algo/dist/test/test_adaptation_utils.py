import pytest
import pytest_parallel
import shutil

import maia
import maia.pytree as PT
import maia.algo.dist.adaptation_utils as adapt_utils
from maia.pytree.yaml          import parse_yaml_cgns

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
def test_duplicate_vtx(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  adapt_utils.duplicate_vtx(zone_n, vtx_pl, comm)

  if comm.rank==0:
    assert PT.Zone.n_vtx(zone_n)==14
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')),    np.array([0,7,14], dtype=np.int32))
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([1.,3.,5.,7.,9., 4.,8.]))
  elif comm.rank==1:
    assert PT.Zone.n_vtx(zone_n)==14
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')),    np.array([7,14,14], dtype=np.int32))
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([2.,4.,6.,8., 3.,7.,6.]))

@pytest_parallel.mark.parallel(2)
def test_remove_vtx(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  adapt_utils.remove_vtx(zone_n, vtx_pl, comm)

  if comm.rank==0:
    assert PT.Zone.n_vtx(zone_n)==4
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')),    np.array([0,3,4], dtype=np.int32))
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([1.,5.,9.]))
  elif comm.rank==1:
    assert PT.Zone.n_vtx(zone_n)==4
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')),    np.array([3,4,4], dtype=np.int32))
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([2.]))

@pytest_parallel.mark.parallel(2)
def test_apply_periodicity_to_vtx(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  periodic = {'translation':np.array([1.,0.,0.])}
  adapt_utils.apply_periodicity_to_vtx(zone_n, vtx_pl, periodic, comm)

  if comm.rank==0:
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([1.,4.,5.,8.,9.]))
  elif comm.rank==1:
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'CoordinateX')), np.array([2.,5.,7.,9.]))

@pytest_parallel.mark.parallel(2)
def test_duplicate_flowsol_elts(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  adapt_utils.duplicate_flowsol_elts(zone_n, vtx_pl-1, 'Vertex', comm)

  if comm.rank==0:
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')), np.array([0,5,9], dtype=np.int32)) # Vertex distrib must be the same
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([1.,3.,5.,7.,9., 4.,8.]))
  elif comm.rank==1:
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')), np.array([5,9,9], dtype=np.int32)) # Vertex distrib must be the same
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([2.,4.,6.,8., 3.,7.,6.]))

@pytest_parallel.mark.parallel(2)
def test_remove_flowsol_elts(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  adapt_utils.remove_flowsol_elts(zone_n, vtx_pl-1, 'Vertex', comm)

  if comm.rank==0:
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')), np.array([0,5,9], dtype=np.int32)) # Vertex distrib must be the same
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([1.,5.,9.]))
  elif comm.rank==1:
    # assert np.array_equal(PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex')), np.array([5,9,9], dtype=np.int32)) # Vertex distrib must be the same
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([2.]))

@pytest_parallel.mark.parallel(2)
def test_apply_periodicity_to_flowsol(comm):
  zone_n = gen_dist_zone(comm)

  if comm.rank==0:
    vtx_pl = np.array(np.array([7,9]), dtype=np.int32)
  elif comm.rank==1:
    vtx_pl = np.array(np.array([2,4,8]), dtype=np.int32)

  periodic = {'rotation_center':np.array([0.,0.,0.]), 'rotation_angle':np.array([0.,0.,np.pi/2.])}
  adapt_utils.apply_periodicity_to_flowsol(zone_n, vtx_pl-1, 'Vertex', periodic, comm)
  periodic = {'rotation_center':np.array([0.,0.,0.]), 'rotation_angle':np.array([0.,0.,-np.pi/2.])}
  adapt_utils.apply_periodicity_to_flowsol(zone_n, vtx_pl-1, 'Vertex', periodic, comm)

  if comm.rank==0:
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([1.,3.,5.,7.,9.]))
  elif comm.rank==1:
    assert np.array_equal(PT.get_value(PT.get_node_from_name(zone_n, 'cX')), np.array([2.,4.,6.,8.]))

def test_get_subset_distribution():
  yz = """
  Zone Zone_t [[1,1,0]]:
    ZoneType ZoneType_t "Unstructured":
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [1]:
      Cell DataArray_t I4 [1]:
    FlowSolVtx FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
    PartialFlowSolFace FlowSolution_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [1]:
    FlowSolCell FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
    PartialFlowSolVtx FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [1]:
    WrongFlowSolCell FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[1]]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yz)
  zone = PT.get_node_from_name(tree, 'Zone')
  node_names     = ['FlowSolVtx','PartialFlowSolFace','FlowSolCell','PartialFlowSolVtx','WrongFlowSolCell']
  expected_names = ['Vertex'    ,'Index',             'Cell',       'Index']
  for node_name, expected_name in zip(node_names, expected_names):
    node = PT.get_node_from_name(zone, node_name)
    distri_n = adapt_utils.get_subset_distribution(zone, node)
    if distri_n is not None:
      assert PT.get_name(distri_n)==expected_name
    else:
      assert PT.get_name(node)=='WrongFlowSolCell'

@pytest_parallel.mark.parallel(2)
def test_convert_vtx_gcs_as_face_bcs(comm):
  if comm.rank==0:
    zone_n = PT.new_Zone('zone', type='Unstructured', size=[[9,3,0]])
    elt_n  = PT.new_Elements('TRI_3', type='TRI_3', erange=np.array([1,3], dtype=np.int32), econn=np.array([1,2,3, 4,5,6], dtype=np.int32), parent=zone_n)
    PT.maia.newDistribution({'Element':[0,2,3]}, parent=elt_n)
    zone_bc_n = PT.new_ZoneBC(parent=zone_n)
    bc_n = PT.new_BC('bc0', point_list=np.array([[3]], dtype=np.int32), loc='FaceCenter', parent=zone_bc_n)
    PT.maia.newDistribution({'Index':[0,1,2]}, parent=bc_n)
    zone_gc_n = PT.new_ZoneGridConnectivity(parent=zone_n)
    gc_n = PT.new_GridConnectivity('gc0', point_list=np.array([[5,7,9]], dtype=np.int32), loc='Vertex', parent=zone_gc_n)
    gc_n = PT.new_GridConnectivity('gc1', point_list=np.array([[1]], dtype=np.int32), loc='Vertex', parent=zone_gc_n)
  elif comm.rank==1:
    zone_n = PT.new_Zone('zone', type='Unstructured', size=[[9,3,0]])
    elt_n  = PT.new_Elements('TRI_3', type='TRI_3', erange=np.array([1,3], dtype=np.int32), econn=np.array([7,8,9], dtype=np.int32), parent=zone_n)
    PT.maia.newDistribution({'Element':[2,3,3]}, parent=elt_n)
    zone_bc_n = PT.new_ZoneBC(parent=zone_n)
    bc_n = PT.new_BC('bc0', point_list=np.array([[2]], dtype=np.int32), loc='FaceCenter', parent=zone_bc_n)
    PT.maia.newDistribution({'Index':[1,2,2]}, parent=bc_n)
    zone_gc_n = PT.new_ZoneGridConnectivity(parent=zone_n)
    gc_n = PT.new_GridConnectivity('gc0', point_list=np.array([[4,6,8]], dtype=np.int32), loc='Vertex', parent=zone_gc_n)
    gc_n = PT.new_GridConnectivity('gc1', point_list=np.array([[2,3]], dtype=np.int32), loc='Vertex', parent=zone_gc_n)

  adapt_utils.convert_vtx_gcs_as_face_bcs(zone_n, {('ZoneGridConnectivity/gc0', 'ZoneGridConnectivity/gc1'):None}, comm)

  # > All GCs are converted, empty GCs shouldn't be transformed
  gc0 = PT.get_node_from_name_and_label(zone_n, 'gc0', 'BC_t')
  gc1 = PT.get_node_from_name_and_label(zone_n, 'gc1', 'BC_t')
  assert gc0 is None
  assert gc1 is not None
  if comm.rank==0:
    assert np.array_equal(PT.get_child_from_name(gc1, 'PointList')[1], np.array([[1]], dtype=np.int32))
  if comm.rank==1:
    assert np.array_equal(PT.get_child_from_name(gc1, 'PointList')[1], np.array([[]], dtype=np.int32))

# @pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
# @pytest_parallel.mark.parallel(2)
# def test_adapt_with_feflo(comm):
