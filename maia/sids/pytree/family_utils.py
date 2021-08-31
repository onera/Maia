import Converter.Internal as I
from maia.sids import pytree as PT
from maia.sids.pytree import predicate as P

def getFamily(parent, family_name):
  try:
    return PT.getNodeFromPredicate(parent, lambda n : P.match_name_label(n, family_name, 'Family_t'))
  except PT.CGNSNodeFromPredicateNotFoundError:
    raise ValueError("Unable to find Family_t with name : {family_name}")

#Next functions are not SIDS compliant, since GC can not have a FamilyName_t node"""

def getGridConnectivitiesFromFamily(parent, family_name):
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromFamily(parent, family_name):
  nodes = []
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromFamily(parent, family_name):
  nodes = []
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      yield node
  return nodes

def belongs_to_family(n, target_family, allow_additional=False):
  family_name_n = PT.requestNodeFromLabel(n, "FamilyName_t", depth=1)
  if family_name_n and I.getValue(family_name_n) == target_family:
    return True
  if allow_additional:
    for additional_family_n in PT.iterNodesFromLabel(n, "AdditionalFamilyName_t", depth=1):
      if I.getValue(additional_family_n) == target_family:
        return True
  return False

def getOneFromLabelsAndFamily(root, labels, family_name):
  predicate = lambda n : I.getType(n) in labels and belongs_to_family(n, family_name)
  node = PT.requestNodeFromPredicate(root, predicate)
  if node:
    return node
  raise ValueError(f"Unable to find node of label {labels} and family name : {family_name}")

def iterFromLabelsAndFamily(root, labels, family_name):
  predicate = lambda n : I.getType(n) in labels and belongs_to_family(n, family_name)
  yield from PT.iterNodesFromPredicate(root, predicate)

def getFromLabelsAndFamily(root, labels, family_name):
  predicate = lambda n : I.getType(n) in labels and belongs_to_family(n, family_name)
  return PT.getNodesFromPredicate(root, predicate)

def getGridConnectivitiesFromAdditionalFamily(parent, family_name):
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  nodes = []
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  nodes = []
  for node in PT.iterNodesFromPredicate(parent, lambda n: I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = PT.requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) == family_name:
      yield node
  return nodes

get_family                  = getFamily
get_grid_connectivities_from_family      = getGridConnectivitiesFromFamily
get_all_grid_connectivities_from_family  = getAllGridConnectivitiesFromFamily
iter_all_grid_connectivities_from_family = getAllGridConnectivitiesFromFamily


