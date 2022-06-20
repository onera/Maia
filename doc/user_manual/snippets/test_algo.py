def test_convert_s_to_u():
  #convert_s_to_u@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree_s = maia.io.file_to_dist_tree(mesh_dir+'/S_twoblocks.yaml', MPI.COMM_WORLD)

  dist_tree_u = maia.algo.dist.convert_s_to_u(dist_tree_s, 'NGON_n', MPI.COMM_WORLD)
  for zone in maia.pytree.get_all_Zone_t(dist_tree_u):
    assert maia.pytree.Zone.Type(zone) == "Unstructured"
  #convert_s_to_u@end

def test_duplicate_zone_with_transformation():
  #duplicate_zone_with_transformation@start
  from mpi4py import MPI
  import maia
  dist_tree = maia.factory.generate_dist_block(10, 'Poly', MPI.COMM_WORLD)
  zone = maia.pytree.get_all_Zone_t(dist_tree)[0]

  dupl_zone = maia.algo.dist.duplicate_zone_with_transformation(zone, "Dupl_zone", translation=[3,0,0])
  #duplicate_zone_with_transformation@end

def test_generate_jns_vertex_list():
  #generate_jns_vertex_list@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree_s = maia.io.file_to_dist_tree(mesh_dir+'/S_twoblocks.yaml', MPI.COMM_WORLD)
  dist_tree = maia.algo.dist.convert_s_to_ngon(dist_tree_s, MPI.COMM_WORLD)

  maia.algo.dist.generate_jns_vertex_list(dist_tree, MPI.COMM_WORLD)
  assert len(maia.pytree.get_nodes_from_name(dist_tree, 'match*#Vtx')) == 2
  #generate_jns_vertex_list@end

def test_merge_zones():
  #merge_zones@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/U_Naca0012_multizone.yaml', MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 3

  maia.algo.dist.merge_zones(dist_tree, ["BaseA/blk1", "BaseB/blk2"], MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 2
  #merge_zones@end

def test_merge_connected_zones():
  #merge_connected_zones@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/U_Naca0012_multizone.yaml', MPI.COMM_WORLD)

  maia.algo.dist.merge_connected_zones(dist_tree, MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 1
  #merge_connected_zones@end

def test_compute_cell_center():
  #compute_cell_center@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  for zone in maia.pytree.iter_all_Zone_t(part_tree):
    cell_center = maia.algo.part.compute_cell_center(zone)
  #compute_cell_center@end

def test_compute_wall_distance():
  #compute_wall_distance@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD, families=["WALL"])
  assert maia.pytree.get_node_from_name(part_tree, "WallDistance") is not None
  #compute_wall_distance@end

def test_interpolate_from_part_trees():
  #interpolate_from_part_trees@start
  import mpi4py
  import numpy
  import maia
  import maia.pytree as PT
  comm = mpi4py.MPI.COMM_WORLD

  dist_tree_src = maia.factory.generate_dist_block(11, 'Poly', comm)
  dist_tree_tgt = maia.factory.generate_dist_block(20, 'Poly', comm)
  part_tree_src = maia.factory.partition_dist_tree(dist_tree_src, comm)
  part_tree_tgt = maia.factory.partition_dist_tree(dist_tree_tgt, comm)
  # Create fake solution
  zone = maia.pytree.get_node_from_label(part_tree_src, "Zone_t")
  src_sol = maia.pytree.newFlowSolution('FlowSolution', 'CellCenter', parent=zone)
  PT.newDataArray("Field", numpy.random.rand(PT.Zone.n_cell(zone)), src_sol)

  maia.algo.part.interpolate_from_part_trees(part_tree_src, part_tree_tgt, comm,\
      ['FlowSolution'], 'Vertex')
  tgt_sol = PT.get_node_from_name(part_tree_tgt, 'FlowSolution')
  assert tgt_sol is not None and PT.Subset.GridLocation(tgt_sol) == 'Vertex'
  #interpolate_from_part_trees@end
