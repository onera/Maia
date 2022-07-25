def test_dist_zone_to_part_zones_only():
  #dist_zone_to_part_zones_only@start
  from mpi4py import MPI
  import os
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import sample_mesh_dir

  comm = MPI.COMM_WORLD
  filename = os.path.join(sample_mesh_dir, 'quarter_crown_square_8.yaml')

  dist_tree = maia.io.file_to_dist_tree(filename, comm)
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_zones = PT.get_all_Zone_t(part_tree)

  include_dict = {'FlowSolution_t' : ['FlowSolution/DataX', 'FlowSolution/DataZ'],
                  'BCDataSet_t'    : ['*']}
  maia.transfer.dist_zone_to_part_zones_only(dist_zone, part_zones, comm, include_dict)

  for zone in part_zones:
    assert PT.get_node_from_path(zone, 'FlowSolution/DataX') is not None
    assert PT.get_node_from_path(zone, 'ZoneSubRegion/Tata') is     None
    assert PT.get_node_from_path(zone, 'ZoneBC/Bnd2/BCDataSet/DirichletData/TutuZ') is not None
  #dist_zone_to_part_zones_only@end

def test_dist_zone_to_part_zones_all():
  #dist_zone_to_part_zones_all@start
  from mpi4py import MPI
  import os
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import sample_mesh_dir

  comm = MPI.COMM_WORLD
  filename = os.path.join(sample_mesh_dir, 'quarter_crown_square_8.yaml')

  dist_tree = maia.io.file_to_dist_tree(filename, comm)
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_zones = PT.get_all_Zone_t(part_tree)

  exclude_dict = {'FlowSolution_t' : ['FlowSolution/DataX', 'FlowSolution/DataZ'],
                  'BCDataSet_t'    : ['*']}
  maia.transfer.dist_zone_to_part_zones_all(dist_zone, part_zones, comm, exclude_dict)

  for zone in part_zones:
    assert PT.get_node_from_path(zone, 'FlowSolution/DataX') is     None
    assert PT.get_node_from_path(zone, 'FlowSolution/DataY') is not None
    assert PT.get_node_from_path(zone, 'ZoneSubRegion/Tata') is not None
    assert PT.get_node_from_path(zone, 'ZoneBC/Bnd2/BCDataSet/DirichletData/TutuZ') is None
  #dist_zone_to_part_zones_all@end

def test_dist_tree_to_part_tree_all():
  #dist_tree_to_part_tree_all@start
  from mpi4py import MPI
  import os
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import sample_mesh_dir

  filename = os.path.join(sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = maia.io.file_to_dist_tree(filename, MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)
  maia.transfer.dist_tree_to_part_tree_all(dist_tree, part_tree, MPI.COMM_WORLD)

  zone = PT.get_all_Zone_t(part_tree)[0]
  assert PT.get_node_from_path(zone, 'FlowSolution/DataX') is not None
  assert PT.get_node_from_path(zone, 'ZoneSubRegion/Tata') is not None
  assert PT.get_node_from_path(zone, 'ZoneBC/Bnd2/BCDataSet/DirichletData/TutuZ') is not None
  #dist_tree_to_part_tree_all@end

def test_dist_tree_to_part_tree_only_labels():
  #dist_tree_to_part_tree_only_labels@start
  from mpi4py import MPI
  import os
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import sample_mesh_dir

  comm = MPI.COMM_WORLD
  filename = os.path.join(sample_mesh_dir, 'quarter_crown_square_8.yaml')

  dist_tree = maia.io.file_to_dist_tree(filename, comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  maia.transfer.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t', 'ZoneSubRegion_t'], comm)

  zone = PT.get_all_Zone_t(part_tree)[0]
  assert PT.get_node_from_path(zone, 'FlowSolution/DataX') is not None
  assert PT.get_node_from_path(zone, 'ZoneSubRegion/Tata') is not None
  assert PT.get_node_from_path(zone, 'ZoneBC/Bnd2/BCDataSet/DirichletData/TutuZ') is None

  maia.transfer.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ['BCDataSet_t'], comm)
  assert PT.get_node_from_path(zone, 'ZoneBC/Bnd2/BCDataSet/DirichletData/TutuZ') is not None
  #dist_tree_to_part_tree_only_labels@end

