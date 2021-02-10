import Converter.Internal as I

from maia.cgns_io                     import cgns_io_tree            as IOT
from maia.transform.transform2        import merge_by_elt_type,\
                                             add_fsdm_distribution,\
                                             gcs_only_for_ghosts
from maia.tree_exchange.tree_transfer import pFlowSolution_to_dFlowSolution

from .               import part                    as PPA
from .load_balancing import setup_partition_weights as DBA


class parallel_tree:
  def __init__(self,comm,dist_tree,part_tree):
    self.comm = comm
    self.part_tree = part_tree
    self.dist_tree = dist_tree

def load_dist_tree(file_name,comm):

  dist_tree = IOT.file_to_dist_tree(file_name, comm, distribution_policy='uniform')

  dzone_to_weighted_parts =  DBA.npart_per_zone(dist_tree, comm, n_part=1)

  merge_by_elt_type(dist_tree,comm) # TODO FSDM-specific

  split_options = {'graph_part_tool' : 'ptscotch', 'save_ghost_data':True,
                   'zone_to_parts':dzone_to_weighted_parts}


def load(file_name,comm):
  dist_tree = load_dist_tree(file_name,comm)
  part_tree = PPA.partitioning(dist_tree, comm, **split_options)

  add_fsdm_distribution(part_tree,comm) # TODO FSDM-specific
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

  add_fsdm_distribution(part_tree,comm) # TODO FSDM-specific
  gcs_only_for_ghosts(part_tree) # TODO FSDM-specific

  return parallel_tree(dist_tree,part_tree)


def merge_and_save(par_tree,file_name):
  # TODO: DiscreteData + BC
  pFlowSolution_to_dFlowSolution(par_tree.dist_tree,par_tree.part_tree,par_tree.comm)
  IOT.dist_tree_to_file(par_tree.dist_tree, file_name, par_tree.comm)
