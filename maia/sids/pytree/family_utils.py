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


#Generation for families

def create_get_from_family(label, family_label):
  def _get_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        return node
    raise ValueError(f"Unable to find {label} from family name : {family_name}")
  return _get_from_family


def create_get_all_from_family(label, family_label):
  def _get_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        nodes.append(node)
    return nodes
  return _get_all_from_family

def create_iter_all_from_family(label, family_label):
  def _iter_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        yield node
  return _iter_all_from_family

for family_label in ['Family_t', 'AdditionalFamily_t']:
  for label in ['Zone_t', 'BC_t', 'ZoneSubRegion_t', 'GridConnectivity_t', 'GridConnectivity1to1_t', 'OversetHoles_t']:
    name = "ToUpdate"  #TODO : update name
    funcname = f"get{label[:-2]}From{family_label[:-2]}"
    func = create_get_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_get_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a list of all CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_iter_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Iterates on CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)


get_family                  = getFamily
get_grid_connectivities_from_family      = getGridConnectivitiesFromFamily
get_all_grid_connectivities_from_family  = getAllGridConnectivitiesFromFamily
iter_all_grid_connectivities_from_family = getAllGridConnectivitiesFromFamily

get_grid_connectivities_from_additional_family      = getGridConnectivitiesFromAdditionalFamily
get_all_grid_connectivities_from_additional_family  = getAllGridConnectivitiesFromAdditionalFamily
iter_all_grid_connectivities_from_additional_family = getAllGridConnectivitiesFromAdditionalFamily
