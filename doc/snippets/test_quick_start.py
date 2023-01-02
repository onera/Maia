import pytest

@pytest.fixture
def convert_yaml():
  from mpi4py import MPI
  import maia.io
  from   maia.utils.test_utils import mesh_dir
  for mesh in ['S_twoblocks', 'U_ATB_45']:
    yaml_name = mesh + '.yaml'
    cgns_name = mesh + '.cgns'
    tree = maia.io.file_to_dist_tree(mesh_dir/yaml_name, MPI.COMM_WORLD)
    maia.io.dist_tree_to_file(tree, cgns_name, MPI.COMM_WORLD)


def test_basic_algo(convert_yaml):
  #basic_algo@start
  from   mpi4py.MPI import COMM_WORLD as comm
  import maia

  tree_s = maia.io.file_to_dist_tree('S_twoblocks.cgns', comm)
  tree_u = maia.algo.dist.convert_s_to_ngon(tree_s, comm)
  maia.io.dist_tree_to_file(tree_u, 'U_twoblocks.cgns', comm)
  #basic_algo@end

def test_workflow(convert_yaml):
  # TODO replace ZSR creation by BC extraction when available
  #workflow@start
  from   mpi4py.MPI import COMM_WORLD as comm
  import maia.pytree as PT
  import maia

  # Read the file. Tree is distributed
  dist_tree = maia.io.file_to_dist_tree('U_ATB_45.cgns', comm)

  # Duplicate the section to a 180° mesh
  # and merge all the blocks into one
  opposite_jns = [['Base/bump_45/ZoneGridConnectivity/matchA'],
                  ['Base/bump_45/ZoneGridConnectivity/matchB']]
  maia.algo.dist.duplicate_from_periodic_jns(dist_tree,
      ['Base/bump_45'], opposite_jns, 22, comm)
  maia.algo.dist.merge_connected_zones(dist_tree, comm)

  # Split the mesh to have a partitioned tree
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  
  # Create a ZSR as well
  for part_zone in PT.get_all_Zone_t(part_tree):
    if PT.get_node_from_name_and_label(part_zone, 'wall', 'BC_t') is not None:
      PT.new_ZoneSubRegion(name='ZSRWall', bc_name='wall', parent=part_zone)

  # Now we can call some partitioned algorithms
  maia.algo.part.compute_wall_distance(part_tree, comm, 
      families=['WALL'], point_cloud='Vertex')
  extract_tree = maia.algo.part.extract_part_from_zsr(part_tree, "ZSRWall", comm)
  slice_tree = maia.algo.part.plane_slice(part_tree, [0,0,1,0], comm,
        containers_name=['WallDistance'])

  # Merge extractions in a same tree in order to save it
  base = PT.get_child_from_label(slice_tree, 'CGNSBase_t')
  PT.set_name(base, f'PlaneSlice')
  PT.add_child(extract_tree, base)
  maia.algo.pe_to_nface(dist_tree,comm)

  extract_tree_dist = maia.factory.recover_dist_tree(extract_tree, comm)
  maia.io.dist_tree_to_file(extract_tree_dist, 'ATB_extract.cgns', comm)
  #workflow@end
  # maia.io.dist_tree_to_file(dist_tree, 'out.cgns', comm) # Write volumic to generate figure

def test_pycgns():
  #pycgns@start
  from   mpi4py.MPI import COMM_WORLD as comm
  import maia
  import Transform.PyTree as CTransform
  import Converter.PyTree as CConverter
  import Post.PyTree      as CPost

  dist_tree = maia.factory.generate_dist_block([101,6,6], 'TETRA_4', comm)
  CTransform._scale(dist_tree, [5,1,1], X=(0,0,0))
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  CConverter._initVars(part_tree, '{Field}=sin({nodes:CoordinateX})')
  part_tree = CPost.computeGrad(part_tree, 'Field')

  maia.transfer.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)
  maia.io.dist_tree_to_file(dist_tree, 'out.cgns', comm)
  #pycgns@end
  # maia.io.write_trees(part_tree, 'part.cgns', comm) # Write partitions to generate figure
  # (crop it h=650)
