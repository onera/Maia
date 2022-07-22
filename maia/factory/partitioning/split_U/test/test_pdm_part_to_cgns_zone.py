import pytest
import numpy as np
from mpi4py import MPI

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.factory.partitioning.split_U import pdm_part_to_cgns_zone as PTC

def test_dump_pdm_output():
  p_zone = I.newZone('Zone.P0.N0', ztype='Unstructured')
  dims = {'n_vtx' : 3}
  data = {'np_vtx_coord' : np.array([1,2,3, 4,5,6, 7,8,9], dtype=np.float64),
          'not_numpy'    : "should_not_be_dumped" }
  PTC.dump_pdm_output(p_zone, dims, data)
  dump_node = I.getNodeFromPath(p_zone, ':CGNS#Ppart')
  assert dump_node is not None
  assert I.getNodeFromName(dump_node, 'n_vtx')[1] == 3
  assert (I.getNodeFromName(dump_node, 'np_vtx_coord')[1] == data['np_vtx_coord']).all()
  assert I.getNodeFromName(dump_node, 'not_numpy') is None

def test_pdm_vtx_to_cgns_grid_coordinates():
  p_zone = I.newZone('Zone.P0.N0', ztype='Unstructured')
  dims = {'n_vtx' : 3}
  data = {'np_vtx_coord' : np.array([1,2,3, 4,5,6, 7,8,9], dtype=np.float64)}

  PTC.pdm_vtx_to_cgns_grid_coordinates(p_zone, dims, data)
  grid_co = I.getNodeFromPath(p_zone, 'GridCoordinates')
  for co in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
    assert I.getNodeFromName1(grid_co, co)[1].dtype == np.float64
  assert (I.getNodeFromName1(grid_co, 'CoordinateX')[1] == [1,4,7]).all()
  assert (I.getNodeFromName1(grid_co, 'CoordinateY')[1] == [2,5,8]).all()
  assert (I.getNodeFromName1(grid_co, 'CoordinateZ')[1] == [3,6,9]).all()

@pytest.mark.parametrize("grid_loc",['FaceCenter', 'Vertex'])
def test_zgc_created_pdm_to_cgns(grid_loc):
  d_zone = I.newZone('ZoneA', ztype='Unstructured')
  p_zone = I.newZone('ZoneA.P0.N0', ztype='Unstructured')
  dims = {}
  data = {'np_face_part_bound_proc_idx' : np.array([0,0,7]),
          'np_face_part_bound_part_idx' : np.array([0,0,7]),
          'np_face_part_bound'          : np.array([1,1,1,97,2,1,1,101,3,1,1,105,4,1,1,109,5,1,1,114,6,1,1,118,7,1,1,119]),
          'np_vtx_part_bound_proc_idx'  : np.array([0,0,16]),
          'np_vtx_part_bound_part_idx'  : np.array([0,0,16]),
          'np_vtx_part_bound'           : np.array([63,1,1,71,71,1,1,63,64,1,1,72,72,1,1,64,65,1,1,73,73,1,1,65,
                                                    66,1,1,74,74,1,1,66,67,1,1,75,75,1,1,67,68,1,1,76,76,1,1,68,
                                                    69,1,1,77,77,1,1,69,70,1,1,78,78,1,1,70])
         }
  PTC.zgc_created_pdm_to_cgns(p_zone, d_zone, dims, data, grid_loc)
  gc_n = I.getNodeFromName(p_zone, 'JN.P0.N0.LT.P1.N0')
  assert I.getValue(gc_n) == 'ZoneA.P1.N0'
  assert I.getValue(I.getNodeFromName(gc_n, 'GridConnectivityType')) == 'Abutting1to1'
  assert PT.Subset.GridLocation(gc_n) == grid_loc
  if grid_loc == 'FaceCenter':
    assert (I.getValue(I.getNodeFromName(gc_n, 'PointList')) == data['np_face_part_bound'][::4]).all()
    assert (I.getValue(I.getNodeFromName(gc_n, 'PointListDonor')) == data['np_face_part_bound'][3::4]).all()
  elif grid_loc == 'Vertex':
    assert (I.getValue(I.getNodeFromName(gc_n, 'PointListDonor')) == data['np_vtx_part_bound'][3::4]).all()

def test_pdm_elmt_to_cgns_elmt_ngon():
  d_zone = I.newZone('Zone', ztype='Unstructured')
  I.newElements('NGonElements', etype='NGON', parent=d_zone) #Dist zone must have a Ngon Node to determine NGon/Element output
  p_zone = I.newZone('Zone.P0.N0', ztype='Unstructured')
  dims = {'n_section' :0, 'n_face' : 6, 'n_cell':1}
  data = {'np_face_cell'     : np.array([1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.int32),
          'np_face_vtx'      : np.array([5, 7, 3, 1, 2, 4, 8, 6, 1, 2, 6, 5, 7, 8, 4, 3, 3,
                                         4, 2, 1, 5, 6, 8, 7], dtype=np.int32),
          'np_face_vtx_idx'  : np.array([0,4,8,12,16,20,24], dtype=np.int32),
          'np_face_ln_to_gn' : np.array([12,5,9,13,18,4], dtype=pdm_dtype),
          'np_cell_face'     : np.array([1,2,3,4,5,6], dtype=np.int32),
          'np_cell_face_idx' : np.array([0,6], dtype=np.int32),
          'np_cell_ln_to_gn' : np.array([42], dtype=pdm_dtype)}

  PTC.pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data)

  ngon_n = I.getNodeFromPath(p_zone, 'NGonElements')
  assert (I.getNodeFromPath(ngon_n, 'ElementStartOffset')[1] == data['np_face_vtx_idx']).all()
  assert (I.getNodeFromPath(ngon_n, 'ElementConnectivity')[1] == data['np_face_vtx']).all()
  assert (I.getNodeFromPath(ngon_n, 'ElementRange')[1] == [1,6]).all()
  assert (I.getVal(MT.getGlobalNumbering(ngon_n, 'Element')) == data['np_face_ln_to_gn']).all()

  nface_n = I.getNodeFromPath(p_zone, 'NFaceElements')
  assert (I.getNodeFromPath(nface_n, 'ElementStartOffset')[1] == data['np_cell_face_idx']).all()
  assert (I.getNodeFromPath(nface_n, 'ElementConnectivity')[1] == data['np_cell_face']).all()
  assert (I.getNodeFromPath(nface_n, 'ElementRange')[1] == [7,7]).all()
  assert (I.getVal(MT.getGlobalNumbering(nface_n, 'Element')) == data['np_cell_ln_to_gn']).all()

def test_pdm_elmt_to_cgns_elmt_elmt():
  d_zone = I.newZone('Zone', ztype='Unstructured')
  I.newElements('NodeB', 'NODE', erange=[5,8], parent=d_zone)
  I.newElements('NodeA', 'NODE', erange=[1,4], parent=d_zone)
  I.newElements('Bar', 'BAR', erange=[9,20], parent=d_zone)
  I.newElements('Quad', 'QUAD', erange=[21,26], parent=d_zone)
  I.newElements('Hexa', 'HEXA', erange=[27,27], parent=d_zone)
  p_zone = I.newZone('Zone.P0.N0', ztype='Unstructured')
  dims = {'n_section' :2, 'n_elt' : [6,1]}
  data = {'0dsections' : [
            {'np_connec' : np.array([3,4], dtype=np.int32), 'np_numabs' : np.array([3,4], dtype=np.int32),
             'np_parent_num' : np.array([3,4]), 'np_parent_entity_g_num' : np.array([3,4])},
            {'np_connec' : np.array([1,2], dtype=np.int32), 'np_numabs' : np.array([1,2], dtype=np.int32),
             'np_parent_num' : np.array([1,2]), 'np_parent_entity_g_num' : np.array([1,2])},
          ],
          '1dsections' : [{'np_connec' : np.array([], dtype=np.int32), 'np_numabs' : np.array([], dtype=np.int32),
                           'np_parent_num' : np.array([]), 'np_parent_entity_g_num' : np.array([])}],
          '2dsections' : [
            {'np_connec' : np.array([1,4,3,2,1,2,6,5,2,3,7,6,3,4,8,7,1,5,8,4,5,6,7,8], dtype=np.int32),
             'np_numabs' : np.array([12,5,9,13,18,4], dtype=np.int32),
             'np_parent_num' : np.array([1,2,3,4,5,6]),
             'np_parent_entity_g_num' : np.array([101,8,6,12,102,103])}
          ],
          '3dsections' : [
            {'np_connec' : np.array([1,2,3,4,5,6,7,8], dtype=np.int32),
             'np_numabs' : np.array([42], dtype=pdm_dtype),
             'np_parent_num' : np.array([1]),
             'np_parent_entity_g_num' : None}
            ]
         }

  PTC.pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data)

  node_n = I.getNodeFromPath(p_zone, 'NodeA')
  assert (I.getNodeFromPath(node_n, 'ElementConnectivity')[1] == data['0dsections'][1]['np_connec']).all()
  assert (I.getNodeFromPath(node_n, 'ElementRange')[1] == [3,4]).all()

  assert I.getNodeFromPath(p_zone, 'Bar') is None

  quad_n = I.getNodeFromPath(p_zone, 'Quad')
  assert (I.getValue(quad_n) == [7,0]).all()
  assert (I.getNodeFromPath(quad_n, 'ElementConnectivity')[1] == data['2dsections'][0]['np_connec']).all()
  assert (I.getNodeFromPath(quad_n, 'ElementRange')[1] == [5,10]).all()
  assert (I.getVal(MT.getGlobalNumbering(quad_n, 'Element')) == data['2dsections'][0]['np_numabs']).all()
  assert (I.getVal(MT.getGlobalNumbering(quad_n, 'Sections')) == data['2dsections'][0]['np_numabs']).all()
  assert (I.getVal(MT.getGlobalNumbering(quad_n, 'ImplicitEntity')) == data['2dsections'][0]['np_parent_entity_g_num']).all()
  assert (I.getVal(I.getNodeFromPath(quad_n, ':CGNS#LocalNumbering/ExplicitEntity')) == \
      data['2dsections'][0]['np_parent_num']).all()

  hexa_n = I.getNodeFromPath(p_zone, 'Hexa')
  assert (I.getValue(hexa_n) == [17,0]).all()
  assert (I.getNodeFromPath(hexa_n, 'ElementConnectivity')[1] == data['3dsections'][0]['np_connec']).all()
  assert (I.getNodeFromPath(hexa_n, 'ElementRange')[1] == [11,11]).all()
  assert (I.getVal(MT.getGlobalNumbering(hexa_n, 'Element')) == data['3dsections'][0]['np_numabs']).all()
  assert (I.getVal(MT.getGlobalNumbering(hexa_n, 'Sections')) == data['3dsections'][0]['np_numabs']).all()
  assert (I.getVal(I.getNodeFromPath(hexa_n, ':CGNS#LocalNumbering/ExplicitEntity')) == \
      data['3dsections'][0]['np_parent_num']).all()

def test_pdm_part_to_cgns_zone():
  # Result of subfunction is not tested here
  d_zone = I.newZone('Zone', ztype='Unstructured')
  I.newElements('Quad', 'QUAD', erange=[2,7], parent=d_zone)
  I.newElements('Hexa', 'HEXA', erange=[1,1], parent=d_zone)
  l_dims = [{'n_section' :2, 'n_cell' : 1, 'n_vtx': 3, 'n_elt' : [6,1]}]
  l_data = [{'0dsections' : [],
            '1dsections' : [],
            '2dsections' : [
              {'np_connec' : np.array([1,4,3,2,1,2,6,5,2,3,7,6,3,4,8,7,1,5,8,4,5,6,7,8], dtype=np.int32),
               'np_numabs' : np.array([12,5,9,13,18,4], dtype=np.int32),
               'np_parent_num' : np.array([1,2,3,4,5,6]),
               'np_parent_entity_g_num' : np.array([101,8,6,12,102,103])}
            ],
           '3dsections' : [
              {'np_connec' : np.array([1,2,3,4,5,6,7,8], dtype=np.int32),
               'np_numabs' : np.array([42], dtype=pdm_dtype),
               'np_parent_num' : np.array([1]),
               'np_parent_entity_g_num' : None}
             ],
           'np_vtx_coord' : np.array([1,2,3, 4,5,6, 7,8,9], dtype=np.float64),
           'np_vtx_part_bound_proc_idx'  : np.array([0,]),
           'np_vtx_part_bound_part_idx'  : np.array([0,]),
           'np_vtx_part_bound'           : np.empty(0, dtype=np.int32),
           'np_vtx_ghost_information'    : np.array([0,0,0,1,1,2]),
           'np_vtx_ln_to_gn'             : np.array([2,4,8,16,64,32]),
           'np_cell_ln_to_gn'            : np.array([42])
           }]

  options = {'part_interface_loc' : 'Vertex', 'dump_pdm_output' : False, 'output_connectivity':'Element'}
  part_zones = PTC.pdm_part_to_cgns_zone(d_zone, l_dims, l_data, MPI.COMM_SELF, options)

  assert len(part_zones) == len(l_dims)
  for ipart, part_zone in enumerate(part_zones):
    assert I.getName(part_zone) == I.getName(d_zone) + '.P0.N{0}'.format(ipart)
    assert (I.getVal(MT.getGlobalNumbering(part_zone, 'Vertex')) == l_data[ipart]['np_vtx_ln_to_gn']).all()
    assert (I.getVal(MT.getGlobalNumbering(part_zone, 'Cell')) == l_data[ipart]['np_cell_ln_to_gn']).all()
