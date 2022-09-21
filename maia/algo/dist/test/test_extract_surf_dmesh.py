import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.pytree        as PT

from maia              import npy_pdm_gnum_dtype as pdm_dtype
from maia.pytree.yaml  import parse_yaml_cgns
from maia.factory      import dcube_generator
from maia.factory      import full_to_dist as F2D

from maia.algo.dist   import extract_surf_dmesh as EXC

@mark_mpi_test(2)
def test_extract_surf_zone(sub_comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], sub_comm)
  zone = PT.get_all_Zone_t(tree)[0]

  #Simplify mesh keeping only 2 BCs
  zone_bc = PT.get_child_from_label(zone, "ZoneBC_t")
  zone_bc[2] = [zone_bc[2][0], zone_bc[2][3]]


  surf_zone = EXC.extract_surf_zone_from_queries(zone, [['ZoneBC_t', 'BC_t']], sub_comm)

  assert PT.Zone.CellSize(surf_zone) == 2*2*2 #2BC, 2*2 faces
  
  dtype = 'I4' if pdm_dtype == np.int32 else 'I8'
  yt = f"""
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t [1,8]:
    ElementStartOffset DataArray_t {dtype} [0,4,8,12,16,20,24,28,32]:
    ElementConnectivity DataArray_t:
      {dtype} : [2,5,4,1,3,6,5,2,5,8,7,4,6,9,8,5,10,11,6,3,11,12,9,6,13,14,11,10,14,15,12,11]
  """
  expected_ngon_full = parse_yaml_cgns.to_node(yt)
  expected_ngon = F2D.distribute_element_node(expected_ngon_full, sub_comm)

  ngon = PT.Zone.NGonNode(surf_zone)
  assert PT.is_same_tree(ngon, expected_ngon)
    

@mark_mpi_test(3)
def test_extract_surf_tree(sub_comm):
  tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
  zone = PT.get_all_Zone_t(tree)[0]

  surf_zone = EXC.extract_surf_zone_from_queries(zone, [['ZoneBC_t', 'BC_t']], sub_comm)
  surf_tree = EXC.extract_surf_tree_from_bc(tree, sub_comm)

  assert len(PT.get_all_CGNSBase_t(surf_tree)) == 1
  assert len(PT.get_all_Zone_t(surf_tree)) == 1
  assert PT.is_same_tree(PT.get_all_Zone_t(surf_tree)[0], surf_zone)
