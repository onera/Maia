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


@pytest.mark.parametrize('elt_name',['BAR_2','TRI_3','TETRA_4'])
@pytest_parallel.mark.parallel([1,2])
def test_remove_elts_from_pl(elt_name, comm):

  dist_tree = maia.factory.dcube_generator.dcube_nodal_generate(3, 1., [0.,0.,0.], 'TETRA_4', comm, get_ridges=True)
  
  # > Define lineic BC
  zone_bc_n = PT.get_node_from_label(dist_tree, 'ZoneBC_t')
  ridge_pl  = np.array([[i for i in range(89,113)]], np.int32)
  ridge_beg =  comm.Get_rank()   *ridge_pl.size//comm.Get_size()
  ridge_end = (comm.Get_rank()+1)*ridge_pl.size//comm.Get_size()
  ridge_pl  = ridge_pl[:,ridge_beg:ridge_end]
  bc_n = PT.new_BC('ridges', point_list=ridge_pl, loc='EdgeCenter', parent=zone_bc_n)
  bc_distri = par_utils.dn_to_distribution(ridge_pl.size, comm)
  PT.maia.newDistribution({'Index':bc_distri}, parent=bc_n)

  # > Define elements to remove
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)==elt_name
  elt_n = PT.get_child_from_predicate(dist_zone, is_asked_elt)
  elt_offset = PT.Element.Range(elt_n)[0]
  if elt_name=='TETRA_4':
    elt_pl = np.array([1,13,2,25,14,37], dtype=np.int32)+1
  elif elt_name=='TRI_3':
    elt_pl = np.array([i for i in range(41,51)], dtype=np.int32)
  else:
    elt_pl = np.array([i for i in range(89,113)], dtype=np.int32)
  i_beg =  comm.Get_rank()   *elt_pl.size//comm.Get_size()
  i_end = (comm.Get_rank()+1)*elt_pl.size//comm.Get_size()
  elt_pl = elt_pl[i_beg:i_end]

  maia.algo.dist.adaptation_utils.remove_elts_from_pl(dist_zone, elt_n, elt_pl, comm)

  # > Checking result
  n_tet = {'TETRA_4':34, 'TRI_3':40, 'BAR_2':40}
  n_tri = {'TETRA_4':48, 'TRI_3':38, 'BAR_2':48}
  n_bar = {'TETRA_4':24, 'TRI_3':24, 'BAR_2': 0}

  is_tet_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4'
  elt_n  = PT.get_child_from_predicate(dist_zone, is_tet_elt)
  elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
  n_elt  = elt_distrib[1]-elt_distrib[0]
  elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
  assert np.array_equal(elt_er, np.array([1, n_tet[elt_name]]))
  assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*4
  assert elt_distrib[2]==n_tet[elt_name]

  cell_distri = par_utils.dn_to_distribution(n_elt, comm)
  assert np.array_equal(PT.maia.getDistribution(dist_zone, 'Cell')[1], cell_distri)

  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3'
  is_tri_bc  = lambda n: PT.get_label(n)=='BC_t'       and PT.Subset.GridLocation(n)=='FaceCenter'
  elt_n  = PT.get_child_from_predicate(dist_zone, is_tri_elt)
  elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
  elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
  n_elt  = elt_distrib[1]-elt_distrib[0]
  expected_er = np.array([n_tet[elt_name]+1, n_tet[elt_name]+n_tri[elt_name]])
  assert np.array_equal(elt_er, expected_er)
  assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*3
  assert elt_distrib[2]==n_tri[elt_name]
  for bc_n in PT.get_nodes_from_predicate(dist_zone, is_tri_bc):
    pl = PT.Subset.getPatch(bc_n)[1]
    assert elt_er[0]<=np.min(pl) and np.max(pl)<=elt_er[1]
  if elt_name=='TRI_3':
    assert PT.get_node_from_name_and_label(dist_zone, 'Zmin', 'BC_t') is None
    zmax_bc_n = PT.get_node_from_name_and_label(dist_zone, 'Zmax', 'BC_t')
    assert PT.maia.getDistribution(zmax_bc_n, 'Index')[1][2]==6

  if elt_name=='BAR_2':
    is_bar_bc  = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='EdgeCenter'
    assert PT.get_child_from_name(dist_zone, 'BAR_2') is None
    assert len(PT.get_nodes_from_predicate(dist_zone, is_bar_bc))==0  
  else: 
    is_bar_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2'
    elt_n  = PT.get_child_from_predicate(dist_zone, is_bar_elt)
    elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
    elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
    n_elt  = elt_distrib[1]-elt_distrib[0]
    expected_er = np.array([n_tet[elt_name]+n_tri[elt_name]+1,
                            n_tet[elt_name]+n_tri[elt_name]+n_bar[elt_name]])
    assert np.array_equal(elt_er, expected_er)
    assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*2
    assert elt_distrib[2]==n_bar[elt_name]


@pytest_parallel.mark.parallel(2)
def test_remove_elts_from_pl_conflict_bc(comm):
  cell_dn = ['[0,1,1]'      ,'[1,1,1]'   ][comm.Get_rank()]
  tri1_ec = ['[1,2,3,3,2,1]','[4,5,6]'   ][comm.Get_rank()]
  tri2_ec = ['[7,8,9,9,8,7]','[10,11,12]'][comm.Get_rank()]
  tri_dn  = ['[0,2,3]'      ,'[2,3,3]'   ][comm.Get_rank()]
  bc_pl   = ['[2,3]'        ,'[6]'       ][comm.Get_rank()]
  bc_dn   = ['[0,2,3]'      ,'[2,3,3]'   ][comm.Get_rank()]

  yt = f"""
    Zone Zone_t:
      ZoneType ZoneType_t 'Unstructured':
      :CGNS#Distribution UserDefinedData_t:
        Cell DataArray_t {cell_dn}:
      TRI_1 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [1, 3]:
        ElementConnectivity DataArray_t I4 {tri1_ec}:
        :CGNS#Distribution UserDefinedData_t:
          Element DataArray_t {tri_dn}:
      TRI_2 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [4, 6]:
        ElementConnectivity DataArray_t I4 {tri2_ec}:
        :CGNS#Distribution UserDefinedData_t:
          Element DataArray_t {tri_dn}:
      TETRA Elements_t I4 [10, 0]:
        ElementRange IndexRange_t I4 [7, 7]:
      ZoneBC ZoneBC_t:
        BC BC_t 'Null':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I4 [{bc_pl}]:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {bc_dn}:
    """

  dist_zone = parse_yaml_cgns.to_node(yt)

  elt_n = PT.get_child_from_name(dist_zone, 'TRI_1')
  elt_pl = np.array([comm.Get_rank()+1], dtype=np.int32)
  maia.algo.dist.adaptation_utils.remove_elts_from_pl(dist_zone, elt_n, elt_pl, comm)

  expected_elt_er = { 'TRI_1': np.array([1,1]),
                      'TRI_2': np.array([2,4]),
                      }
  expected_elt_dn = { 'TRI_1': np.array([[0,0,1],[0,1,1]][comm.rank]),
                      'TRI_2': np.array([[0,2,3],[2,3,3]][comm.rank]),
                      }
  for elt_name, elt_er in expected_elt_er.items():
    elt_n = PT.get_child_from_name(dist_zone, elt_name)
    assert np.array_equal(PT.Element.Range(elt_n), elt_er)
    assert np.array_equal(PT.maia.getDistribution(elt_n, 'Element')[1], expected_elt_dn[elt_name])
  
  expected_bc_pl = np.array([[[1]],[[4]]][comm.rank])
  expected_bc_dn = np.array([[0,1,2],[1,2,2]][comm.rank])
  bc_n = PT.get_node_from_label(dist_zone, 'BC_t')
  assert np.array_equal(PT.Subset.getPatch(bc_n)[1], expected_bc_pl)
  assert np.array_equal(PT.maia.getDistribution(bc_n, 'Index')[1], expected_bc_dn)

  expected_cell_dn = np.array([[0,1,1],[1,1,1]][comm.rank])
  assert np.array_equal(PT.maia.getDistribution(dist_zone, 'Cell')[1], expected_cell_dn)


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
