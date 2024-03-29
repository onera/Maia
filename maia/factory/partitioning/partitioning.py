import numpy as np
import time
from mpi4py import MPI

import maia.pytree as PT

from maia import pdm_has_ptscotch, pdm_has_parmetis
from maia.algo.dist import matching_jns_tools     as MJT
from maia.algo.part import connectivity_transform as CNT
from maia.utils     import par_utils
from maia.utils     import logging as mlog

from maia.transfer.dist_to_part import data_exchange  as BTP

from .load_balancing import setup_partition_weights as SPW
from .split_S import part_zone      as partS
from .split_U import part_all_zones as partU
from .post_split import post_partitioning as post_split
from .load_balancing import balancing_quality

def set_default(dist_tree, comm):

  default_renum = {'cell_renum_method' : 'NONE',
                   'face_renum_method' : 'NONE',
                   'vtx_renum_method'  : 'NONE',
                   'n_cell_per_cache'  : 0,
                   'n_face_per_pack'   : 0,
                   'graph_part_tool'   : None }

  default = {'graph_part_tool'         : None,
             'zone_to_parts'           : None,
             'reordering'              : default_renum,
             'part_interface_loc'      : 'Vertex',
             'output_connectivity'     : 'Element',
             'preserve_orientation'    : False,
             'save_all_connectivities' : False,
             'additional_ln_to_gn'     : [],
             'keep_empty_sections'     : False,
             'dump_pdm_output'         : False }

  if pdm_has_parmetis:
    default['graph_part_tool'] = 'parmetis'
  elif pdm_has_ptscotch:
    default['graph_part_tool'] = 'ptscotch'
  else:
    default['graph_part_tool'] = 'hilbert'
  default['reordering']['graph_part_tool'] = default['graph_part_tool']

  # part_interface_loc : Vertex si Elements, FaceCenter si NGons
  for zone in PT.get_all_Zone_t(dist_tree):
    if 22 in [PT.Element.Type(elt) for elt in PT.iter_children_from_label(zone, 'Elements_t')]:
      default['part_interface_loc'] = 'FaceCenter'
      break

  return default

def partition_dist_tree(dist_tree, comm, **kwargs):
  """Perform the partitioning operation: create a partitioned tree from the input distributed tree.

  The input tree can be structured or unstuctured, but hybrid meshes are not yet supported.

  Important:
    Geometric information (such as boundary conditions, zone subregion, etc.) are reported
    on the partitioned tree; however, data fields (BCDataSet, FlowSolution, etc.) are not
    transfered automatically. See ``maia.transfer`` module.

  See reference documentation for the description of the keyword arguments.

  Args:
    dist_tree (CGNSTree): Distributed tree
    comm      (MPIComm) : MPI communicator
    **kwargs  : Partitioning options
  Returns:
    CGNSTree: partitioned cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #partition_dist_tree@start
        :end-before: #partition_dist_tree@end
        :dedent: 2
  """

  options = set_default(dist_tree, comm)
  subkeys = ['reordering'] #Key for which we have sub dicts

  # > Check if input keys exist
  for key, val in kwargs.items():
    if key in options.keys():
      if key in subkeys:
        assert isinstance(val, dict)
        for subkey, subval in val.items():
          if not subkey in options[key].keys():
            mlog.error(f'Partitioning sub keyword "{key}/{subkey}" does not exists and will be ignored')
    else:
      mlog.error(f'Partitioning keyword "{key}" does not exists and will be ignored')
  # > Erase default setting with user settings
  for key, val in kwargs.items():
    if key in options.keys():
      if not key in subkeys:
        options[key] = val
      else:
        for subkey, subval in val.items():
          if subkey in options[key].keys():
            options[key][subkey] = subval
  # > Check some values
  assert options['graph_part_tool'] in partU.maia_to_pdm_split_tool
  # TODO we should rename this part_tool because not all methods involve a graph
  assert options['part_interface_loc'] in ['Vertex', 'FaceCenter']
  assert options['output_connectivity'] in ['Element', 'NGon']

  # > Setup balanced weight if no provided
  zone_to_parts = options.pop('zone_to_parts')
  if zone_to_parts is None:
    zone_to_parts = SPW.balance_multizone_tree(dist_tree, comm)
  assert isinstance(zone_to_parts, dict)
  # > Call main function
  part_tree = _partitioning(dist_tree, zone_to_parts, comm, options)
  
  # Compute statistics
  if not par_utils.any_true(zone_to_parts.values(), lambda e: len(e)>1, comm):
    # Rebuild array 
    zone_paths = PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')
    n_cell_per_block = np.zeros(len(zone_paths), np.int32)
    for part_zone_path in PT.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
      part_zone = PT.get_node_from_path(part_tree, part_zone_path)
      idx = zone_paths.index(PT.maia.conv.get_part_prefix(part_zone_path))
      n_cell = PT.Zone.n_cell(part_zone) # If zone is a point cloud, use n_vtx
      n_cell_per_block[idx] = n_cell if n_cell > 0 else PT.Zone.n_vtx(part_zone)
    if comm.Get_rank() == 0:
      mlog.stat("[partition_dist_tree] After partitioning, repartition statistics are:")

    balancing_quality.compute_balance_and_splits(n_cell_per_block, comm, comm.Get_rank()==0)

  return part_tree

def _partitioning(dist_tree,
                  dzone_to_weighted_parts,
                  comm,
                  part_options):

  n_blocks = len(PT.get_all_Zone_t(dist_tree))
  blocks_str = "blocks" if n_blocks > 1 else "block"
  mlog.info(f"Partitioning tree of {n_blocks} initial {blocks_str}...")
  start = time.time()
  is_s_zone = lambda n : PT.get_label(n) == 'Zone_t' and PT.Zone.Type(n) == 'Structured'
  is_u_zone = lambda n : PT.get_label(n) == 'Zone_t' and PT.Zone.Type(n) == 'Unstructured'

  MJT.add_joins_donor_name(dist_tree, comm)

  part_tree = PT.new_CGNSTree()
  dist_zones_S = []
  part_zones_S = []
  for dist_base in PT.iter_all_CGNSBase_t(dist_tree):

    part_base = PT.new_node(PT.get_name(dist_base), 'CGNSBase_t', PT.get_value(dist_base), parent=part_tree)
    #Add top level nodes
    for node in PT.get_children(dist_base):
      if PT.get_label(node) != "Zone_t":
        PT.add_child(part_base, PT.deep_copy(node))

    #Split S zones : we create a subcom for each zone, to avoid serialization of part_zone
    sub_comms = []
    for zone in PT.iter_children_from_predicate(dist_base, is_s_zone):
      zone_path = PT.get_name(dist_base) + '/' + PT.get_name(zone)
      weights = dzone_to_weighted_parts.get(zone_path, [])
      sub_comms.append(comm.Split(len(weights)>0))

    for zone, sub_comm in zip(PT.iter_children_from_predicate(dist_base, is_s_zone), sub_comms):
      zone_path = PT.get_name(dist_base) + '/' + PT.get_name(zone)
      weights = dzone_to_weighted_parts.get(zone_path, [])
      if len(weights) > 0:
        s_parts = partS.part_s_zone(zone, weights, sub_comm, comm.Get_rank())
        for part in s_parts:
          PT.add_child(part_base, part)
      else:
        s_parts = []
      part_zones_S.append(s_parts)
      dist_zones_S.append(zone)

  # Transfert coords for S zones, all at once to avoid multiple block_to_parts
  BTP.dist_coords_to_part_coords_m(dist_zones_S, part_zones_S, comm)


  all_s_parts = PT.get_all_Zone_t(part_tree) #At this point we only have S parts
  partS.split_original_joins_S(all_s_parts, comm)

  #Split U zones (all at once)
  base_to_blocks_u = {PT.get_name(base) : [zone for zone in PT.get_all_Zone_t(base) if is_u_zone(zone)] \
      for base in PT.get_all_CGNSBase_t(dist_tree)}
  has_u_zones = any([values != [] for values in base_to_blocks_u.values()])
  if has_u_zones:
    base_to_parts_u = partU.part_U_zones(base_to_blocks_u, dzone_to_weighted_parts, comm, part_options)
    for base, u_parts in base_to_parts_u.items():
      part_base = PT.get_child_from_name(part_tree, base)
      for u_part in u_parts:
        if not part_options['preserve_orientation']:
          CNT.enforce_boundary_pe_left(u_part)
        PT.add_child(part_base, u_part)

  post_split(dist_tree, part_tree, comm)
  end = time.time()
  n_cell     = sum([PT.Zone.n_cell(zone) for zone in PT.iter_all_Zone_t(part_tree)])
  n_cell_all = comm.allreduce(n_cell, MPI.SUM)
  mlog.info(f"Partitioning completed ({end-start:.2f} s) -- "
            f"Nb of cells for current rank is {mlog.size_to_str(n_cell)} "
            f"(Σ={mlog.size_to_str(n_cell_all)})")

  return part_tree
