from maia.cgns_io            import load_collective_size_tree       as LST
from maia.cgns_io            import cgns_io_tree                    as IOT
from maia.cgns_io.hdf_filter import tree                            as HTF
from maia.connectivity       import generate_ngon_from_std_elements as FTH
from maia.partitioning       import part                            as PPA
import maia.distribution.distribution_tree                          as MDI
from maia.cgns_registry      import cgns_registry                   as CGR
from maia.cgns_registry      import cgns_keywords
from maia.cgns_registry      import tree                            as CGT # Not bad :D
from maia.transform.transform2 import merge_by_elt_type, add_fsdm_distribution, gcs_only_for_ghosts
from maia.cgns_io import save_part_tree as SPT
from maia.tree_exchange.tree_transfer import pFlowSolution_to_dFlowSolution

from Converter import cgnskeywords as CGK
import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY
import sys

from maia.connectivity.generate_ngon_from_std_elements import generate_ngon_from_std_elements




class parallel_tree:
  def __init__(self,comm,dist_tree,part_tree):
    self.comm = comm
    self.part_tree = part_tree
    self.dist_tree = dist_tree



def load_partitioned_tree(file_name,comm):
  # > Load only the list of zone and sizes ...
  dist_tree = LST.load_collective_size_tree(file_name, comm)

  cgr = CGT.add_cgns_registry_information(dist_tree, comm)

  # I.printTree(dist_tree)
  # exit(2)
  # > ParaDiGM : dcube_gen() --> A faire

  MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')
  # I.printTree(dist_tree)

  hdf_filter = dict()
  HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

  # for key, val in hdf_filter.items():
  #   print(key, val)

  # skip_type_ancestors = ["Zone_t/FlowSolution_t/"]
  # skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun"], ["ZoneSubRegion_t", "VelocityY"]]
  # skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "*"], ["Zone_t", "ZoneSubRegion_t", "VelocityY"]]
  skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "Momentum*"],
                         ["Zone_t", "ZoneSubRegion_t", "Velocity*"]]
  # hdf_filter_wo_fs = IOT.filtering_filter(dist_tree, hdf_filter, skip_type_ancestors, skip=True)
  # # IOT.load_tree_from_filter(file_name, dist_tree, comm, hdf_filter)

  # for key, val in hdf_filter_wo_fs.items():
  #   print(key, val)
  # IOT.load_tree_from_filter(file_name, dist_tree, comm, hdf_filter_wo_fs)
  IOT.load_tree_from_filter(file_name, dist_tree, comm, hdf_filter)
  #generate_ngon_from_std_elements(dist_tree,comm)
  #C.convertPyTree2File(dist_tree, "dist_tree_ngon.cgns")

  #hdf_filter = dict()
  #HTF.create_tree_hdf_filter(dist_tree, hdf_filter, mode='write')
  ##print("hdf_filter = ",hdf_filter)
  #IOT.save_tree_from_filter("dist_tree_0.hdf", dist_tree, comm, hdf_filter)

  # FTH.generate_ngon_from_std_elements(dist_tree, comm)

  # C.convertPyTree2File(dist_tree, "dist_tree_{0}.hdf".format(rank))

  # I.printTree(dist_tree)
  # > To copy paste in new algorithm
  # dzone_to_proc = compute_distribution_of_zones(dist_tree, distribution_policy='uniform', comm)
  # > dzone_to_weighted_parts --> Proportion de la zone initiale qu'on souhate après partitionnement
  # > dloading_procs        --> Proportion de la zone initiale avant le partitionnement (vision block)
  #
  # > ... and this is suffisent to predict your partitions sizes

  #dzone_to_weighted_parts = DBA.computePartitioningWeights(dist_tree, comm) # TODO use this
  dzone_to_weighted_parts = {}
  for zone in I.getZones(dist_tree):
      dzone_to_weighted_parts[zone[0]] = [1./comm.Get_size()]

  # print(dzone_to_weighted_parts)

  dloading_procs = dict()
  for zone in I.getZones(dist_tree):
    dloading_procs[zone[0]] = list(range(comm.Get_size()))
  # print(dloading_procs)

  merge_by_elt_type(dist_tree,comm) # TODO FSDM-specific

  #hdf_filter = dict()
  #HTF.create_tree_hdf_filter(dist_tree, hdf_filter)
  #IOT.save_tree_from_filter("dist_tree_bef_0.hdf", dist_tree, comm, hdf_filter)

  part_tree = PPA.partition_by_elt(dist_tree,comm,split_method=2)

  add_fsdm_distribution(part_tree,comm) # TODO FSDM-specific
  gcs_only_for_ghosts(part_tree) # TODO FSDM-specific

  ## TODO
  #for zone in I.getZones(part_tree):
  #  fs_n = I.newFlowSolution(name="FlowSolution#EndOfRun", gridLocation='Vertex', parent=zone)
  #  vtx_gi_n = I.getNodeFromName(zone, "np_vtx_ghost_information")
  #  I.newDataArray("GhostInfo", vtx_gi_n[1], parent=fs_n)

  return parallel_tree(comm,dist_tree,part_tree)

def load_partitioned_tree_poly(file_name,comm):
  dist_tree = LST.load_collective_size_tree(file_name, comm)

  cgr = CGT.add_cgns_registry_information(dist_tree, comm)

  MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

  hdf_filter = dict()
  HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

  skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "Momentum*"],
                         ["Zone_t", "ZoneSubRegion_t", "Velocity*"]]

  IOT.load_tree_from_filter(file_name, dist_tree, comm, hdf_filter)

  #dzone_to_weighted_parts = DBA.computePartitioningWeights(dist_tree, comm) # TODO use this
  dzone_to_weighted_parts = {}
  for zone in I.getZones(dist_tree):
      dzone_to_weighted_parts[zone[0]] = [1./comm.Get_size()]

  dloading_procs = dict()
  for zone in I.getZones(dist_tree):
    dloading_procs[zone[0]] = list(range(comm.Get_size()))

  #merge_by_elt_type(dist_tree,comm) # TODO FSDM-specific

  part_tree = PPA.partition(dist_tree,comm,split_method=2)

  add_fsdm_distribution(part_tree,comm) # TODO FSDM-specific
  gcs_only_for_ghosts(part_tree) # TODO FSDM-specific

  return parallel_tree(dist_tree,part_tree)


def merge_and_save(par_tree,file_name):
  # TODO: DiscreteData + BC
  pFlowSolution_to_dFlowSolution(par_tree.dist_tree,par_tree.part_tree,par_tree.comm)
  IOT.dist_tree_to_file(par_tree.dist_tree, file_name, par_tree.comm)
