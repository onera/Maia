import Converter.Internal as I

from maia.cgns_io                     import cgns_io_tree            as IOT
from maia.transform.transform        import merge_by_elt_type,\
                                            gcs_only_for_ghosts

from maia.tree_exchange.utils import get_partitioned_zones
from maia.tree_exchange.part_to_dist.data_exchange import part_sol_to_dist_sol

from maia.partitioning import part as PPA
from maia.partitioning.load_balancing import setup_partition_weights as DBA


class parallel_tree:
  def __init__(self,comm,dist_tree,part_tree):
    self.comm = comm
    self.part_tree = part_tree
    self.dist_tree = dist_tree

def load_dist_tree(file_name,comm):

  dist_tree = IOT.file_to_dist_tree(file_name, comm, distribution_policy='uniform')

  merge_by_elt_type(dist_tree,comm) # TODO FSDM-specific

  return  dist_tree


def load(file_name,comm):
  dist_tree = load_dist_tree(file_name,comm)

  dzone_to_weighted_parts =  DBA.npart_per_zone(dist_tree, comm, n_part=1)
  split_options = {'graph_part_tool' : 'ptscotch', 'save_ghost_data':True,
                   'zone_to_parts':dzone_to_weighted_parts}

  part_tree = PPA.partitioning(dist_tree, comm, **split_options)

  gcs_only_for_ghosts(part_tree) # TODO FSDM-specific

  ## TODO
  #for zone in I.getZones(part_tree):
  #  fs_n = I.newFlowSolution(name="FlowSolution#EndOfRun", gridLocation='Vertex', parent=zone)
  #  vtx_gi_n = I.getNodeFromName(zone, "np_vtx_ghost_information")
  #  I.newDataArray("GhostInfo", vtx_gi_n[1], parent=fs_n)

  return parallel_tree(comm,dist_tree,part_tree)

def load_partitioned_tree_poly(file_name,comm):

  dist_tree = IOT.file_to_dist_tree(file_name, comm, distribution_policy='uniform')
  dzone_to_weighted_parts =  DBA.npart_per_zone(dist_tree, comm, n_part=1)

  #merge_by_elt_type(dist_tree,comm) # TODO FSDM-specific

  split_options = {'graph_part_tool' : 'ptscotch', 'save_ghost_data':True,
                   'zone_to_parts':dzone_to_weighted_parts}
  #part_tree = PPA.partition(dist_tree,comm,split_method=2)
  part_tree = PPA.partitioning(dist_tree, comm, **split_options)

  gcs_only_for_ghosts(part_tree) # TODO FSDM-specific

  return parallel_tree(dist_tree,part_tree)


def merge_and_save(par_tree, file_name):
  # TODO: BC
  for d_base in I.getBases(par_tree.dist_tree):
    for d_zone in I.getZones(d_base):
      p_zones = get_partitioned_zones(par_tree.part_tree, I.getName(d_base) + '/' + I.getName(d_zone))
      part_sol_to_dist_sol(d_zone, p_zones, par_tree.comm)

  IOT.dist_tree_to_file(par_tree.dist_tree, file_name, par_tree.comm)
