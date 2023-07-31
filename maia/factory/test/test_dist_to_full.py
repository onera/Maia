import pytest 
import pytest_parallel

from mpi4py import MPI

import maia
import maia.pytree        as PT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.factory import dist_to_full

def test_reshape_S_arrays():
  yt = """
  Zone Zone_t [[3,2,0], [3,2,0], [2,1, 0]]:
    ZoneType ZoneType_t "Structured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    FlowSolution FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Sol DataArray_t [1,2,3,4]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  dist_to_full._reshape_S_arrays(tree)
  assert (PT.get_node_from_name(tree, 'Sol')[1] == [ [[1],[3]], [[2],[4]] ]).all()
  assert (PT.get_node_from_name(tree, 'CoordinateX')[1] == \
          [ [[1,10], [4,13], [7,16]], [[2,11],[5,14],[8,17]], [[3,12],[6,15], [9,18]] ]).all()

@pytest_parallel.mark.parallel(2)
def test_dist_to_full_tree_U(comm):
  dist_tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  full_tree = dist_to_full.dist_to_full_tree(dist_tree, comm, target=1)

  if comm.Get_rank() == 0:
    assert full_tree is None
  else:
    assert PT.get_node_from_name(dist_tree, ':CGNS#Distribution') #Distri info has been removed
    tree = maia.factory.generate_dist_block(3, 'HEXA_8', MPI.COMM_SELF)
    PT.rm_nodes_from_name(tree, ':CGNS#Distribution') #Remove to compare
    assert PT.is_same_tree(tree, full_tree)

@pytest_parallel.mark.parallel(2)
def test_dist_to_full_tree_S(comm):
  dist_tree = maia.factory.generate_dist_block(3, 'S', comm)
  full_tree = dist_to_full.dist_to_full_tree(dist_tree, comm, target=1)
  if comm.Get_rank() == 0:
    assert full_tree is None
  else:
    for array in PT.get_nodes_from_label(full_tree, 'DataArray_t'):
      assert array[1].shape == (3,3,3) #Only coords in this mesh

