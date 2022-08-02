import Converter.Internal as I
import maia.pytree as PT

from maia import pdm_has_ptscotch, pdm_has_parmetis
from maia.algo.dist import matching_jns_tools     as MJT
from maia.algo.part import connectivity_transform as CNT

from .load_balancing import setup_partition_weights as SPW
from .split_S import part_zone      as partS
from .split_U import part_all_zones as partU
from .post_split import post_partitioning as post_split


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
             'additional_connectivity' : [],
             'additional_ln_to_gn'     : [],
             'additional_color'        : [],
             'dump_pdm_output'    : False }

  if pdm_has_parmetis:
    default['graph_part_tool'] = 'parmetis'
  elif pdm_has_ptscotch:
    default['graph_part_tool'] = 'ptscotch'
  else:
    default['graph_part_tool'] = 'hilbert'
  default['reordering']['graph_part_tool'] = default['graph_part_tool']

  # part_interface_loc : Vertex si Elements, FaceCenter si NGons
  for zone in I.getZones(dist_tree):
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
            print('Warning -- Unvalid subkey {0}/{1} in partitioning'.format(key,subkey))
    else:
      print('Warning -- Unvalid key {0} in partitioning'.format(key))
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
  assert options['graph_part_tool'] in ['ptscotch', 'parmetis', 'hilbert', None]
  assert options['part_interface_loc'] in ['Vertex', 'FaceCenter']
  assert options['output_connectivity'] in ['Element', 'NGon']

  # > Setup balanced weight if no provided
  zone_to_parts = options.pop('zone_to_parts')
  if zone_to_parts is None:
    zone_to_parts = SPW.balance_multizone_tree(dist_tree, comm)
  assert isinstance(zone_to_parts, dict)
  # > Call main function
  return _partitioning(dist_tree, zone_to_parts, comm, options)

def _partitioning(dist_tree,
                  dzone_to_weighted_parts,
                  comm,
                  part_options):

  u_zones   = [zone for zone in I.getZones(dist_tree) if PT.Zone.Type(zone) == 'Unstructured']
  s_zones   = [zone for zone in I.getZones(dist_tree) if PT.Zone.Type(zone) == 'Structured']

  if len(u_zones)*len(s_zones) != 0:
    raise RuntimeError("Hybrid meshes are not yet supported")

  MJT.add_joins_donor_name(dist_tree, comm)

  is_s_zone = lambda n : I.getType(n) == 'Zone_t' and PT.Zone.Type(n) == 'Structured'
  is_u_zone = lambda n : I.getType(n) == 'Zone_t' and PT.Zone.Type(n) == 'Unstructured'

  part_tree = I.newCGNSTree()
  for dist_base in PT.iter_all_CGNSBase_t(dist_tree):

    part_base = I.createNode(I.getName(dist_base), 'CGNSBase_t', I.getValue(dist_base), parent=part_tree)
    #Add top level nodes
    for node in I.getChildren(dist_base):
      if I.getType(node) != "Zone_t":
        I.addChild(part_base, node)

    #Split S zones
    for zone in PT.iter_children_from_predicate(dist_base, is_s_zone):
      zone_path = I.getName(dist_base) + '/' + I.getName(zone)
      weights = dzone_to_weighted_parts.get(zone_path, [])
      s_parts = partS.part_s_zone(zone, weights, comm)
      for part in s_parts:
        I._addChild(part_base, part)

  all_s_parts = I.getZones(part_tree) #At this point we only have S parts
  partS.split_original_joins_S(all_s_parts, comm)

  #Split U zones (all at once)
  base_to_blocks_u = {I.getName(base) : I.getZones(base) for base in I.getBases(dist_tree)}
  if len(u_zones) > 0:
    base_to_parts_u = partU.part_U_zones(base_to_blocks_u, dzone_to_weighted_parts, comm, part_options)
    for base, u_parts in base_to_parts_u.items():
      part_base = I.getNodeFromName1(part_tree, base)
      for u_part in u_parts:
        if not part_options['preserve_orientation']:
          try:
            PT.Zone.NGonNode(u_part)
            CNT.enforce_boundary_pe_left(u_part)
          except RuntimeError: #Zone is elements-defined
            pass
        I._addChild(part_base, u_part)

  post_split(dist_tree, part_tree, comm)

  return part_tree
