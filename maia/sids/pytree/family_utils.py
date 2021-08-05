# Keys to access TreeNode values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3
# --------------------------------------------------------------------------
#
#   getFamily, getAdditionalFamily
#
# --------------------------------------------------------------------------
def getFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in getNodesFromLabel(parent, 'Family_t'):
    if I.getName(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find Family_t with name : {family_name}")

def getAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in getNodesFromLabel(parent, 'Family_t'):
    if I.getName(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find Family_t with name : {family_name}")


# --------------------------------------------------------------------------
def getGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      yield node
  return nodes

# --------------------------------------------------------------------------
def getGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      yield node
  return nodes

get_family                  = getFamily
get_grid_connectivities_from_family      = getGridConnectivitiesFromFamily
get_all_grid_connectivities_from_family  = getAllGridConnectivitiesFromFamily
iter_all_grid_connectivities_from_family = getAllGridConnectivitiesFromFamily

get_grid_connectivities_from_additional_family      = getGridConnectivitiesFromAdditionalFamily
get_all_grid_connectivities_from_additional_family  = getAllGridConnectivitiesFromAdditionalFamily
iter_all_grid_connectivities_from_additional_family = getAllGridConnectivitiesFromAdditionalFamily
