import Converter.Internal as I
import maia.sids.sids     as SIDS

from maia import pdm_has_ptscotch, pdm_has_parmetis
from maia.transform.dist_tree import add_joins_ordinal      as AJO
from .split_S import part_zone      as partS
from .split_U import part_all_zones as partU
from .post_split import post_partitioning as post_split

def set_default(dist_tree, comm):

  default_renum = {'cell_renum_method' : 'NONE',
                   'face_renum_method' : 'NONE',
                   'n_cell_per_cache'  : 0,
                   'n_face_per_pack'   : 0,
                   'graph_part_tool'   : None }

  default = {'graph_part_tool'    : None,
             'zone_to_parts'      : {I.getName(z):[1/comm.Get_size()] for z in I.getZones(dist_tree)},
             'reordering'         : default_renum,
             'part_interface_loc' : 'Vertex',
             'output_connectivity': 'Element',
             'dump_pdm_output'    : False }

  if pdm_has_ptscotch:
    default['graph_part_tool'] = 'ptscotch'
  elif pdm_has_parmetis:
    default['graph_part_tool'] = 'parmetis'
  default['reordering']['graph_part_tool'] = default['graph_part_tool']

  # part_interface_loc : Vertex si Elements, FaceCenter si NGons
  for zone in I.getZones(dist_tree):
    if 22 in [SIDS.ElementType(elt) for elt in I.getNodesFromType1(zone, 'Elements_t')]:
      default['part_interface_loc'] = 'FaceCenter'
      break

  return default

def partitioning(dist_tree, comm, **kwargs):

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
  assert options['graph_part_tool'] in ['ptscotch', 'parmetis', None]
  assert options['part_interface_loc'] in ['Vertex', 'FaceCenter']
  assert options['output_connectivity'] in ['Element', 'NGon']
  assert isinstance(options['zone_to_parts'], dict)

  # > Call main function
  zone_to_parts = options.pop('zone_to_parts')
  return _partitioning(dist_tree, zone_to_parts, comm, options)

def _partitioning(dist_tree,
                  dzone_to_weighted_parts,
                  comm,
                  part_options):

  all_zones = I.getZones(dist_tree)
  u_zones   = [zone for zone in all_zones if SIDS.Zone.Type(zone) == 'Unstructured']
  s_zones   = [zone for zone in all_zones if SIDS.Zone.Type(zone) == 'Structured']

  if len(u_zones)*len(s_zones) != 0:
    raise RuntimeError("Hybrid meshes are not yet supported")

  if I.getNodeFromName(dist_tree, 'OrdinalOpp') is None:
    AJO.add_joins_ordinal(dist_tree, comm)

  part_tree = I.newCGNSTree()
  #For now only one base
  dist_base = I.getNodeFromType1(dist_tree, 'CGNSBase_t')
  part_base = I.createNode(I.getName(dist_base), 'CGNSBase_t', I.getValue(dist_base), parent=part_tree)

  #Split S zones
  all_s_parts = []
  for zone in s_zones:
    s_parts = partS.part_s_zone(zone, dzone_to_weighted_parts[I.getName(zone)], comm)
    for part in s_parts:
      I._addChild(part_base, part)
    all_s_parts.extend(s_parts)
  partS.split_original_joins_S(all_s_parts, comm)

  #Split U zones
  if len(u_zones) > 0:
    u_parts = partU.part_U_zones(u_zones, dzone_to_weighted_parts, comm, part_options)
    for part in u_parts:
      I._addChild(part_base, part)

  post_split(dist_base, part_base, comm)

  #Add top level nodes
  for node in I.getChildren(dist_base):
    if I.getType(node) != "Zone_t":
      I.addChild(part_base, node)

  return part_tree
