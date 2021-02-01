import Converter.Internal as I
import maia.sids.sids     as SIDS

from maia.transform.dist_tree import add_joins_ordinal      as AJO
from .split_S import part_zone      as partS
from .split_U import part_all_zones as partU

def partitioning(dist_tree,
                 dzone_to_weighted_parts,
                 comm,
                 split_options   = None,
                 reorder_options = None):

  all_zones = I.getZones(dist_tree)
  u_zones   = [zone for zone in all_zones if SIDS.ZoneType(zone) == 'Unstructured']
  s_zones   = [zone for zone in all_zones if SIDS.ZoneType(zone) == 'Structured']

  if len(u_zones)*len(s_zones) != 0:
    raise RuntimeError("Hybrid meshes are not yet supported")

  if I.getNodeFromName(dist_tree, 'OrdinalOpp') is None:
    AJO.add_joins_ordinal(dist_tree, comm)

  part_tree = I.newCGNSTree()
  #For now only one base
  dist_base = I.getNodeFromType1(dist_tree, 'CGNSBase_t')
  part_base = I.createNode(I.getName(dist_base), 'CGNSBase_t', I.getValue(dist_base), parent=part_tree)

  #Split S zones
  for zone in s_zones:
    s_parts = partS.part_s_zone(zone, dzone_to_weighted_parts[I.getName(zone)], comm)
    for part in s_parts:
      I._addChild(part_base, part)

  #Split U zones
  if len(u_zones) > 0:
    #Manage defaults
    if split_options is None:
      split_options = dict()
    split_options.setdefault('split_method', 'parmetis')
    split_options.setdefault('no_weight', False)
    split_options.setdefault('jn_location', 'FaceCenter')
    split_options.setdefault('save_ghost_data', False)

    u_parts = partU.part_U_zones(u_zones, dzone_to_weighted_parts, comm, split_options, reorder_options)
    for part in u_parts:
      I._addChild(part_base, part)

  #Add top level nodes
  for fam in I.getNodesFromType1(dist_base, 'Family_t'):
    I.addChild(part_base, fam)

  return part_tree
