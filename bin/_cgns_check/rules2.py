import difflib
import sys
import inspect
import collections

import maia.pytree      as PT
import maia.pytree.pred as PTp

from maia.pytree.typing import List, CGNSTree # Strangly import * brings NamedTuple as a function

from maia.pytree.cgns_keywords import dtype_to_cgns

from .data import LABEL_PROPS, ALL_LABELS, UNITS_ENUM, UNITS_NAME, DATANAME_IDENTIFIERS

OK = ''

class DictDifflibCache(dict):
    def __init__(self, possibilities, n, cutoff):
        self.possibilities = possibilities
        self.n = n
        self.cutoff = cutoff
    def __missing__(self, key):
        val = difflib.get_close_matches(key, self.possibilities, self.n, self.cutoff)
        self[key] = val
        return val

# Errors code 200-299
# 201 - 210 : global consistency & references
# 211 - 220 : node attributes
# 221 - 230 : child structure
# 231 - 240 : zone topology
# 241 - 250 : subsets
# 251 - 260 : fields 

difflib_cache = DictDifflibCache(DATANAME_IDENTIFIERS, 1, 0.75)

def sibling_zones_inttype_consistency(nodes:List[CGNSTree]) -> str: 
    """E201 - Sibling zones integer type consistency
    
    All the Zone_t nodes found under a same CGNSBase_t parent should
    have the same integer datatype (I4 or I8).
    
    Erroneous tree example:

    CGNSTree CGNSTree_t:
    └───Base CGNSBase_t I4 [3 3]
        ├───Rotor Zone_t \033[91mI8\033[0m [[72254 60288  0]]
        └───Stator Zone_t \033[91mI4\033[0m [[72254 60288  0]]
    """
    last = nodes[-1]
    if PT.get_label(last) != 'CGNSBase_t':
        return OK
    
    ztypes = {zone[0] : dtype_to_cgns[zone[1].dtype] for zone in 
              PT.iter_all_Zone_t(last) if zone[1] is not None}

    if len(set(ztypes.values())) > 1:
        first = next(iter(ztypes.keys()))
        other = next(key for key in ztypes if ztypes[key] != ztypes[first])
        return f"Integer type of zone '{first}' is {ztypes[first]},"  \
               f" but integer type of zone '{other}' is {ztypes[other]}"
    else:
        return OK

def zone_inttype_consistency(nodes:List[CGNSTree]) -> str: 
    """E202 - Zone integer type consistency
    
    Within a given Zone_t node, connectivity arrays must have the same integer datatype
    than the one of the zone (I4 or I8).
    The connectivity arrays are:
        - ElementRange, ElementStartOffset, ElementConnectivity, ParentElements under Elements_t nodes
        - IndexRange_t and IndexArray_t nodes
    
    Erroneous tree example:

    Stator Zone_t \033[32mI8\033[0m [[72254 60288  0]]
    ├───NGonElements Elements_t I4 [22 0]
    │   ├───ElementRange IndexRange_t \033[32mI8\033[0m [1 192556]
    │   ├───ElementStartOffset DataArray_t \033[32mI8\033[0m (192557,)
    │   ├───ElementConnectivity DataArray_t \033[91mI4\033[0m (770224,)
    │   └───ParentElements DataArray_t \033[32mI8\033[0m (192556, 2)
    └───ZoneBC ZoneBC_t
        ├───bc_moyeu.4 BC_t "FamilySpecified"
        │   └───PointList IndexArray_t \033[91mI4\033[0m (1, 864)
        └───bc_carter.5 BC_t "FamilySpecified"
            └───PointList IndexArray_t \033[32mI8\033[0m (1, 1632)
    """
    #TODO : maybe call directly on relevant nodes instead of zone
    last = nodes[-1]

    elt_names = ['ElementConnectivity', 'ElementStartOffset', 'ParentElements']
    check = PT.get_label(last) in ['IndexRange_t', 'IndexArray_t'] \
         or PT.get_name(last) in elt_names and PT.get_label(nodes[-2]) == 'Elements_t'

    if check:
        ztype = PT.get_value_type(nodes[2])
        if (ntype := PT.get_value_type(last)) != 'MT' and ntype != ztype:
            return f"Datatype of connectivity data ({ntype}) is not equal to zone datatype ({ztype})"

    return OK

def missing_family(nodes:List[CGNSTree]) -> str: 
    """E206 - Missing Family_t node

    Families referenced by an (Additional)FamilyName_t node
    must be defined as Family_t in the CGNSBase_t.

    Erroneous tree example:

    Base CGNSBase_t I4 [3 3]
    ├───FARFIELD Family_t
    │╴╴╴\033[91mMissing 'WALL' Family_t node\033[0m
    └───Stator Zone_t I4 [[72254 60288  0]]
        └───ZoneBC ZoneBC_t:
            └───bc_142 BC_t "BCWall"
                ├───GridLocation GridLocation_t "FaceCenter"
                ├───PointList IndexArray_t I4 (1, 15168)
                └───FamilyName FamilyName_t \033[32m"WALL"\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) in ['FamilyName_t', 'AdditionalFamilyName_t']:
        value = PT.get_str_value(last)
        base = nodes[1]
        if PT.get_child_from_name_and_label(base, value, 'Family_t') is None:
            return f"Referenced family '{value}' is not defined at base level"
    return OK

def family_without_familybc(nodes:List[CGNSTree]) -> str: 
    """E207 - Missing FamilyBC_t node

    If a BC_t node is of type "FamilySpecified", a FamilyBC_t child must
    exists below the corresponding Family_t node.

    Erroneous tree example:

    Base CGNSBase_t I4 [3 3]
    ├───\033[32mWALL\033[0m Family_t
    │   └╴╴╴\033[91mMissing FamilyBC_t node\033[0m
    └───Stator Zone_t I4 [[72254 60288  0]]
        └───ZoneBC ZoneBC_t:
            └───bc_142 BC_t \033[32m"FamilySpecified"\033[0m
                ├───GridLocation GridLocation_t "FaceCenter"
                ├───PointList IndexArray_t I4 (1, 15168)
                └───FamilyName FamilyName_t \033[32m"WALL"\033[0m
    """
    # Check for Family_t nodes or BC_t nodes ?
    # Check on Family is less efficient but display error only once
    last = nodes[-1]
    if PT.get_label(last) != 'Family_t' or PT.get_child_from_label(last, 'FamilyBC_t') is not None:
        return OK

    fam = PT.get_name(last)
    base = nodes[-2]
    rel_bc = PTp.label_is('BC_t') & PTp.value_is('FamilySpecified') & PTp.belongs_to_family(fam, False)
    bc = PT.get_child_from_predicates(base, ['Zone_t', 'ZoneBC_t', rel_bc])
    if bc is not None:
        return f"Family is referenced by at least one BC, but FamilyBC_t child is missing"
    return OK

def missing_zsr_subset(nodes:List[CGNSTree]) -> str:
    """E208 - Missing ZSR related node

    When a ZoneSubRegion_t defines its related subset through
    a BCRegionName or a GridConnectivityRegionName child,
    the related subset must exists in tree.

    Erroneous tree example:

    Stator Zone_t I4 [[72254 60288  0]]
    ├───ZoneBC ZoneBC_t:
    │   └───bc_142 BC_t "BCWall"
    └───ZSR ZoneSubRegion_t
        └───BCRegionName Descriptor_t \033[91m"bc_146" # BC does not exists in tree\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) == 'ZoneSubRegion_t' and not PT.Container._is_subset(last):
        try:
            PT.Container.SubsetNodePath(last, nodes[-2])
        except ValueError:
            linked_names = ['BCRegionName', 'GridConnectivityRegionName']
            refnode = PT.find_child_from_predicate(last, PTp.name_in(linked_names))
            return f"ZoneSubRegion_t related subset {PT.get_str_value(refnode)} does not exists"

    return OK

def gc_opposite_zone(nodes:List[CGNSTree]) -> str: 
    """E209 - Missing opposite zone

    For GridConnectivity(1to1)_t nodes, the opposite zone name
    registered in node value must lead to an existing zone.

    Erroneous tree example:

    CME2 CGNSBase_t I4 [3 3]
    ├───row_1_down Zone_t I4 [[21 20 0] [85 84 0] [141 140 0]]
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t
    │       ├───gc_1_005 GridConnectivity1to1_t \033[91m"row_1_upStream"\033[0m
    │       └───gc_2_005 GridConnectivity1to1_t \033[32m"row_1_inlet"\033[0m
    ├───\033[32mrow_1_inlet\033[0m Zone_t I4 [[17 16 0] [85 84 0] [17 16 0]]
    ╵╴╴╴\033[91mMissing row_1_upStream zone\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) in ['GridConnectivity_t', 'GridConnectivity1to1_t']:
        base_name = PT.get_name(nodes[1])
        opp_path = PT.GridConnectivity.ZoneDonorPath(last, base_name)
        if PT.get_node_from_path(nodes[0], opp_path) is None:
            return f"Target zone of GC_t node not found in tree: '{PT.get_value(last)}'"
    return OK



def invalid_name(nodes:List[CGNSTree]) -> str:
    """E211 - Invalid node name

    Name of nodes must be shorten than 32 characters and should contain
    only ascii characters.

    Erroneous tree example:

    FlowSolution FlowSolution_t
    ├───GridLocation GridLocation_t "CellCenter"
    └───\033[91msource_term_rans(turbulence_closure)\033[0m DataArray_t (7200,)  \033[91m# Name is too long\033[0m
    """
    last = nodes[-1]
    name = PT.get_name(last)
    if len(name) > 32:
        return f"Maximal len for node name is 32"
    if not name.isascii():
        return f"Name contains non ascii characters"
    if '/' in name:
        return f"Name should not contain '/' character"
    return OK

def invalid_label(nodes:List[CGNSTree]) -> str: 
    """E212 - Invalid label
    
    Each node must have a label that is listed among the allowed
    labels.

    Erroneous tree example:

    AUBE BC_t "FamilySpecified"
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───PointList \033[91mArray_t\033[0m I4 (1, 15168) \033[91m# Not in labels list\033[0m
    └───FamilyName FamilyName_t "WALL"

    """
    label = PT.get_label(nodes[-1])
    if label not in ALL_LABELS:
        return f"Invalid label {label}"
    return OK

def unexpected_value(nodes:List[CGNSTree]) -> str:
    """E213 - Unexpected value for MT node

    Some label have 'MT' kind, so corresponding nodes should have no
    value.

    Erroneous tree example:

    row_2_flux_1_Main_Blade_inlet Zone_t I4 [[17 16 0] [85 85 0] [21 20 0]]
    ├───ZoneType ZoneType_t "Structured"
    └───GridCoordinates GridCoordinates_t \033[91m"Cartesian" # Node should be MT\033[0m
    """
    last = nodes[-1]
    label = PT.get_label(last)
    props = LABEL_PROPS[label]
    if props.get('TYPE', '') == 'MT':
        if PT.get_value(last) is not None:
            return f"Unexpected value for {label} node, which should be MT"
    return OK

def missing_value(nodes:List[CGNSTree]) -> str:
    """E214 - Missing value for non-MT node

    Some label have non-'MT' kind, so corresponding nodes should have
    a value.

    Erroneous tree example:

    row_2_flux_1_Main_Blade_inlet Zone_t I4 [[17 16 0] [85 85 0] [21 20 0]]
    ├───ZoneType ZoneType_t \033[91m # Missing 'C1' value \033[0m
    └───GridCoordinates GridCoordinates_t
    """
    last = nodes[-1]
    label = PT.get_label(last)
    props = LABEL_PROPS[label]
    if props.get('TYPE', 'MT') != 'MT':
        if PT.get_value(last) is None and label != 'ZoneSubRegion_t':
            return f"Missing value for {label} node, which should be of kind {props['TYPE']}"
    return OK

def invalid_datatype(nodes:List[CGNSTree]) -> str:
    """E215 - Invalid data type

    Some nodes require their value to have a specific type.

    Erroneous tree example:

    ATB91 CGNSBase_t I4 [3 3]
    └───zone Zone_t I4 [[11193 7200 0]]
        ├───ZoneType ZoneType_t "Unstructured"
        └───NGonElements Elements_t \033[91mR8\033[0m [22 0] \033[91m# Type should be I4\033[0m
    """
    # Note: DataArray_t is not treated in this generic func because kind can be almost eveything
    last = nodes[-1]
    label = PT.get_label(last)
    props = LABEL_PROPS[label]
    expt_type = props.get('TYPE', 'MT')
    if label == 'IndexArray_t':
        expt_type = 'R' if PT.get_name(last) == 'InwardNormalList' else 'I'
    if expt_type != 'MT':
        if (type:=PT.get_value_type(last)) != 'MT':
            if type[:len(expt_type)] != expt_type: # Cut to compare if expt_type is only 'I' or 'R'
                return f"Invalid datatype for {label} node: expected {expt_type}, got {type}"
    return OK

def invalid_datashape(nodes:List[CGNSTree]) -> str:
    """E216 - Invalid data shape

    Some nodes require their value to have a specific shape.

    Erroneous tree example:

    ATB91 CGNSBase_t I4 [3 3]
    └───zone Zone_t I4 [[11193 7200 0]]
        ├───ZoneType ZoneType_t "Unstructured"
        └───NGonElements Elements_t I4 \033[91m[22]   # Shape should be (2,)\033[0m
    """
    # Note: DataArray_t is not treated in this generic func because shape can be almost eveything
    last = nodes[-1]
    label = PT.get_label(last)
    props = LABEL_PROPS[label]
    expt_shape = props.get('SHAPE', None)
    # Compute expected shape for these labels (depens on idx_dim)
    if label in ['IndexRange_t', 'Rind_t', 'Zone_t']:
        idx_dim = 1 if PT.Zone.Type(nodes[2]) == 'Unstructured' else PT.get_np_value(nodes[1])[0]
        if label == 'IndexRange_t':
            expt_shape = (2,) if PT.get_label(nodes[-2]) == 'Elements_t' else (idx_dim, 2)
        elif label == 'Rind_t':
            expt_shape = (2*idx_dim,)
        elif label == 'Zone_t':
            expt_shape = (idx_dim, 3)

    if expt_shape is not None:
        if (val:=PT.get_value(last, True)) is not None:
            shape = val.shape
            if shape != expt_shape:
                return f"Invalid shape for {label} node: expected {expt_shape}, got {shape}"

    # Deal IndexArray_t : only first dim can be checked
    if label == 'IndexArray_t':
        if (val:=PT.get_value(last, True)) is not None:
            shape = val.shape
            if PT.get_name(last) == 'InwardNormalList':
                expt = PT.Zone.PhysicalDimension(nodes[2])
            elif PT.get_name(last) in ['PointListDonor', 'CellListDonor']:
                opp_zone_path = PT.GridConnectivity.ZoneDonorPath(nodes[-2], PT.get_name(nodes[1]))
                opp_zone = PT.find_node_from_path(nodes[0], opp_zone_path)
                expt = PT.Zone.IndexDimension(opp_zone)
            else:
                expt = PT.Zone.IndexDimension(nodes[2])
            if shape[0] != expt:
                return f"Invalid shape for IndexArray_t node: expected ({expt}, N), got {shape}"

    return OK

def invalid_datavalue(nodes:List[CGNSTree]) -> str:
    """E217 - Invalid data value

    Some nodes, especially terminal enumerated nodes, require their value
    to belong to a specific set.

    Erroneous tree example:

    ATB91 CGNSBase_t I4 [3 3]
    └───zone Zone_t I4 [[11193 7200 0]]
        ├───ZoneType ZoneType_t \033[91m"Polyedric" # Invalid value for ZoneType_t enum\033[0m
        └───NGonElements Elements_t I4 [22 0]
    """
    # Note: DataArray_t is not treated in this generic func because shape can be almost eveything
    last = nodes[-1]
    label = PT.get_label(last)
    props = LABEL_PROPS[label]
    expt_value = props.get('ALLOWED_VALUE', None)

    if expt_value is not None:
        if (value:=PT.get_value(last)) is not None:
            if value not in expt_value:
                return f"'{value}' is not a admissible value for a {label} node"
    if PT.get_label(last) in ['DimensionalUnits_t', 'AdditionalUnits_t']:
        expt_list  = UNITS_ENUM[:5] if PT.get_label(last) == 'DimensionalUnits_t' else UNITS_ENUM[5:]
        units_name = UNITS_NAME[:5] if PT.get_label(last) == 'DimensionalUnits_t' else UNITS_NAME[5:]
        if (value:=PT.get_value(last)) is not None:
            for i, (v, expt) in enumerate(zip(value, expt_list)):
                if v not in expt:
                    name = f"{units_name[i]}Units"
                    return f"'{v}' is not a admissible value for {i+1}th field ({name}) of {label} node"
        
    return OK




def duplicated_children_name(nodes:List[CGNSTree]) -> str:
    """E221 - Duplicated children name

    Two children of the same parent node can not have
    the same name.

    Erroneous tree example:

    Wing BC_t "BCWall"
    ├───\033[91mFamilyName\033[0m FamilyName_t "WALL"
    └───\033[91mFamilyName\033[0m AdditionalFamilyName_t "AIRPLANE" \033[91m# 'FamilyName' already used\033[0m
    """
    last = nodes[-1]
    counting = collections.defaultdict(int)
    for child in PT.get_children(last):
        counting[PT.get_name(child)] += 1
    for key, val in counting.items():
        if val > 1:
            return f"Duplicated name '{key}' ({val} occurences found)"
    return OK

def unexpected_child_label(nodes:List[CGNSTree]) -> str: 
    """E222 - Unexpected child label
    
    Each child must have a label that is listed among the allowed
    labels for the current node.
    This list depends of the label of the input node.

    Erroneous tree example:

    AUBE \033[32mBC_t\033[0m "FamilySpecified"
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───PointList IndexArray_t I4 (1, 15168)
    ├───DirichletData \033[91mBCData_t #Not allowed here\033[0m
    └───FamilyName FamilyName_t "WALL"

    Correct example:

    AUBE \033[32mBC_t\033[0m "FamilySpecified"
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───PointList IndexArray_t I4 (1, 15168)
    ├───BCDataSet \033[32mBCDataSet_t\033[0m "Null"
    │   └───DirichletData \033[32mBCData_t #OK\033[0m
    └───FamilyName FamilyName_t "WALL"
    """
    if len(nodes) < 2:
        return OK

    parent_label = PT.get_label(nodes[-2])
    node_label = PT.get_label(nodes[-1])
    
    parent_props = LABEL_PROPS[parent_label]
    parent_admissible_labels = (n[0] for n in parent_props.get('ALLOWED_CHILDREN', []))
    if node_label not in parent_admissible_labels:
        return f"Child of label {node_label} is not allowed under a {parent_label} parent"
    else:
        return OK

def reserved_child_name(nodes:List[CGNSTree]) -> str: 
    """E223 - Use of reserved name
    
    Some nodes have a list of names which are reserved for specific
    children labels.

    Erroneous tree example:

    ZoneBC \033[32mZoneBC_t\033[0m
    ├───wing BC_t "BCWall"
    ├───fuselage BC_t "BCWall"
    ├───\033[91mReferenceState\033[0m BC_t "BCWall" \033[91m# Name is reserved for ReferenceState_t node\033[0m
    └───farfield BC_t "BCFarfield"
    """
    if len(nodes) < 2:
        return OK

    nname = PT.get_name(nodes[-1])
    nlabel = PT.get_label(nodes[-1])
    plabel = PT.get_label(nodes[-2])

    parent_props = LABEL_PROPS[plabel]
    parent_reserved_names = parent_props["RESERVED_NAMES"]
    if nname in parent_reserved_names:
        if (lbl:=parent_reserved_names[nname]) is None:
            return f"Usage of '{nname}' within {plabel} structure is forbidden"
        elif nlabel != lbl:
            return f"Within parent structure {plabel}, name '{nname}' can not be used for a {nlabel} node"

    return OK

def predefined_child_name(nodes:List[CGNSTree]) -> str: 
    """E224 - Predefined name
    
    Some nodes have a list of children for which the name
    can not be freely choosen.

    Erroneous tree example:

    ZoneBC \033[32mZoneBC_t\033[0m
    ├───\033[91mRefState\033[0m ReferenceState_t \033[91m# Name of ReferenceState_t must be ReferenceState\033[0m
    ├───wing BC_t "BCWall"
    ├───fuselage BC_t "BCWall"
    └───farfield BC_t "BCFarfield"
    """
    if len(nodes) < 2:
        return OK

    nname = PT.get_name(nodes[-1])
    nlabel = PT.get_label(nodes[-1])
    plabel = PT.get_label(nodes[-2])
    
    parent_props = LABEL_PROPS[plabel]
    for lbl in parent_props.get('ALLOWED_CHILDREN', []):
        if nlabel == lbl[0]:
            if len(lbl) > 2:
                expt = lbl[2]
                if isinstance(expt, str) and nname != expt:
                    return f"Within parent structure {plabel}, name of {nlabel} node should be '{expt}'"
                # NB: some labels have imposed nodes + 'userdefined', symbolized by '*'
                elif isinstance(expt, list) and '*' not in expt and nname not in expt:
                    return f"Within parent structure {plabel}, name of {nlabel} node should be one of {expt}"
            break

    return OK


def too_many_children(nodes:List[CGNSTree]) -> str:
    """E225 - Too many children of same label
    
    Each node has a list of allowed children labels; for some of
    them, the number of occurences is limited.

    Erroneous tree example:

    bc_142 BC_t "FamilySpecified"
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───PointList IndexArray_t I4 (1, 15168)
    ├───FamilyName FamilyName_t "WALL"
    └───SecondFamilyName \033[91mFamilyName_t\033[0m "AUBE" \033[91m# At most one FamilyName_t allowed\033[0m

    """
    last = nodes[-1]
    label = PT.get_label(last)

    try:
        allowed = LABEL_PROPS[label]['ALLOWED_CHILDREN']
    except KeyError:
        return OK # Invalid label OR no allowed children --> other rules catch it

    counting = collections.defaultdict(int)
    for child in PT.get_children(last):
        counting[PT.get_label(child)] += 1
    for lbl in allowed:
        name = lbl[0]
        card = lbl[1]
        max_allowed = -1 # Negative means unlimited
        if isinstance(card, int):
            max_allowed = card
        elif card == '?':
            max_allowed = 1
        elif card.startswith('<='):
            max_allowed = int(card[2:])
        if not max_allowed < 0:
            cnt = counting[name]
            if cnt > max_allowed:
                return f"Too many children of label {name}: expected at most {max_allowed}, got {cnt}"

    return OK

def too_few_children(nodes:List[CGNSTree]) -> str:
    """E226 - Missing children of specific label
    
    Each node has a list of allowed children labels; for some of
    them, the presence of children node(s) is mandatory.

    Erroneous tree example:

    Stator Zone_t I4 [[72254 60288  0]]
    │╴╴╴\033[91mMissing ZoneType_t node\033[0m
    ├───GridCoordinates GridCoordinates
    ├───ZoneBC ZoneBC_t
    └───FlowSolution#Init FlowSolution_t
    """
    last = nodes[-1]
    label = PT.get_label(last)

    try:
        allowed = LABEL_PROPS[label]['ALLOWED_CHILDREN']
    except KeyError:
        return OK # Invalid label OR no allowed children --> other rules catch it

    counting = collections.defaultdict(int)
    for child in PT.get_children(last):
        counting[PT.get_label(child)] += 1
    for data in allowed:
        lbl = data[0]
        card = data[1]
        min_required = 0
        if isinstance(card, int):
            min_required = card
        elif card.startswith('>='):
            min_required = int(card[2:])
        cnt = counting[lbl]
        if cnt < min_required:
            if min_required == 1:
                return f"Missing required child of label {lbl}"
            else:
                return f"Too few children of label {lbl}: expected at least {min_required}, got {cnt}"

    # Also treat in this rule special case of EXACTLY_ONE_OF with only one item => required child
    for data in LABEL_PROPS[label].get('EXACTLY_ONE_OF', []):
        if len(data) == 1 and PT.get_child_from_name(last, data[0]) is None:
            return f"Missing required child of name {data[0]}"

    return OK

def exclusive_children(nodes:List[CGNSTree]) -> str:
    """E227 - Mutually exclusive children

    Some node require to choose between two (or more) possible
    children, but in an exclusive way.

    Erroneous tree example:

    Wing BC_t "BCWall"
    ├───\033[91mPointList\033[0m IndexRange_t I4 [[31 40]]
    └───\033[91mPointRange\033[0m IndexArray_t I4 (1, 10) \033[91m# Only of PointList or PointRange is allowed\033[0m
    """
    last = nodes[-1]
    try:
        data = LABEL_PROPS[PT.get_label(last)]["EXACTLY_ONE_OF"]
        exclusive = [d for d in data if len(d) > 1]
    except KeyError:
        return OK

    for group in exclusive:
        found = [PT.get_child_from_name(last, name) is not None for name in group]
        if sum(found) < 1:
            return f"Exactly one child among {group} is required, but none were found"
        elif sum(found) > 1:
            # Special error message if len==2
            if len(group) == 2:
                return f"Exactly one child among {group} is required, but both were found"
            fnd_names = tuple(g for g,f in zip(group, found) if f)
            return f"Exactly one child among {group} is required, but several were found: {fnd_names}"

    return OK





def zone_cell_dimension(nodes:List[CGNSTree]) -> str: 
    """E231 - Zone cell dimension
    
    The maximal dimension of Elements_t nodes of a given Zone_t must be consistent with the cell
    dimension of its CGNSBase parent.
    
    Erroneous tree example:

    FlatPlate CGNSBase_t I4 [\033[32m2\033[0m 3]
    └───Zone Zone_t I4 [[876 1633 0]]
        ├───TRI_3 Elements_t I4 [\033[32m5\033[0m 0]     \033[32m#Surfacic elements\033[0m
        └───TETRA_4 Elements_t I4 [\033[91m10\033[0m 0]  \033[91m#Volumic elements\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK

    base = nodes[-2]
    assert PT.get_label(base) == 'CGNSBase_t'
    cell_dim = PT.get_np_value(base)[0]
    
    #Skip if zone has mixed elements (check will be done in phase 3)
    zone_elts = set(PT.Element.Type(e) for e in PT.iter_children_from_label(last, 'Elements_t'))
    if 'MIXED' in zone_elts:
        return OK

    if (zdim := PT.Zone.CellDimension(last)) != cell_dim:
        return f"Maximal dimension of zone elements not consistent with cell dimension of the parent base:" \
               f" expected {cell_dim}, got {zdim}"

    return OK

def zone_physical_dimension(nodes:List[CGNSTree]) -> str: 
    """E232 - Zone physical dimension
    
    The number of coordinates arrays of a Zone_t node must be consistent with the physical
    dimension of its CGNSBase parent.
    
    Erroneous tree example:

    FlatPlate CGNSBase_t I4 [2 \033[32m3\033[0m]
    └───Zone Zone_t I4 [[876 1633 0]]
        └───GridCoordinates GridCoordinates_t
            ├───CoordinateX DataArray_t R8 (876,)
            ├───CoordinateY DataArray_t R8 (876,)
            ╵╴╴╴\033[91mMissing CoordinateZ array\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) != 'GridCoordinates_t':
        return OK

    base = nodes[-3]
    assert PT.get_label(base) == 'CGNSBase_t'
    phy_dim = PT.get_np_value(base)[1]

    pred = PTp.label_is('DataArray_t') & ~PTp.name_is('CoordinateTransform')
    zdim = len(PT.get_children_from_predicate(last, pred))
    if zdim != phy_dim:
        return f"Number of coordinates array is not consistent with physical dimension of the parent base:" \
            f" expected {phy_dim}, got {zdim}"
    return OK

def S_zone_size(nodes:List[CGNSTree]) -> str:
    """E233 - Size of structured zone

    For structured zones, the number of cells and vertices in each direction
    must be consistent.

    Erroneous tree example:

    row_2_flux_1_Main_Blade_inlet Zone_t I4 [[\033[32m17 16\033[0m  0] [\033[91m85 85\033[0m  0] [\033[32m21 20\033[0m  0]]
    └───ZoneType ZoneType_t "Structured"
    """
    last = nodes[-1]
    if PT.get_label(last) == 'Zone_t' and PT.Zone.Type(last) == 'Structured':
        vtx_size = PT.Zone.VertexSize(last)
        cell_size = PT.Zone.CellSize(last)
        if any(v != c + 1 for v,c in zip(vtx_size, cell_size)):
            return f"Inconsistent number of vertices {vtx_size} and cells {cell_size} for structured zone"
        if any(v == 1 for v in vtx_size):
            return f"At least one direction have no cells; a mesh of lower " \
                   f"dimension should be used instead (CellSize={cell_size})"

        return OK
    return OK

def zone_elements_mixup(nodes:List[CGNSTree]) -> str:
    """E234 - Incompatible elements type

    The elements sections belonging to a same Zone_t node
    must be either polyedric elements or standard elements,
    but not a mix of the two.

    Erroneous tree example:

    Zone Zone_t I4 [[1000 2187 0]]
    ├───NFaceElements Elements_t I4 [23 0]           \033[32m#Polyedric\033[0m
    │   └───ElementRange IndexRange_t I4 [   1 2187]
    ├───TRI Elements_t I4 [5 0]                      \033[91m#Standard\033[0m
    │   └───ElementRange IndexRange_t I4 [2188 2673]
    └───QUAD Elements_t I4 [ 7 0]                    \033[91m#Standard\033[0m
        └───ElementRange IndexRange_t I4 [2674 2916]
    """
    last = nodes[-1]
    if not PT.get_label(last) == 'Zone_t':
        return OK
    
    elts = {PT.Element.Type(e) for e in PT.iter_children_from_label(nodes[-1], 'Elements_t')}
    if 'NFACE_n' in elts or 'NGON_n' in elts:
        poly = elts & {'NFACE_n', 'NGON_n'}
        std = elts - {'NFACE_n', 'NGON_n', 'BAR_2', 'NODE'}
        if len(std) > 0:
            return f"Standard sections {std} and polyedric sections {poly} can not be used together"
 
    return OK

def zone_number_of_elements(nodes:List[CGNSTree]) -> str: 
    """E235 - Number of native mesh elements
    
    The total number of mesh elements whose dimension equals CellDimension
    must be equal to the number of elements of the Zone_t node.
    
    Erroneous tree example:

    Zone Zone_t I4 [[876 \033[32m1633\033[0m 0]]
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [1 \033[91m1600\033[0m]
    └───BAR Elements_t I4 [3 0]  
        └───ElementRange IndexRange_t I4 [1601 1716]
    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t' or PT.Zone.Type(last) == 'Structured':
        return OK

    #Skip if zone has mixed elements (check will be done in phase 3)
    zone_elts = set(PT.Element.Type(e) for e in PT.iter_children_from_label(last, 'Elements_t'))
    if 'MIXED' in zone_elts:
        return OK
    # Also skip polyedric zones with only PE (check will be done in phase 3)
    if PTp.IS_POLY2D_ZONE(last):
        if not PT.Zone.has_ngon_elements(last):
            return OK
    elif PTp.IS_POLY3D_ZONE(last):
        if not PT.Zone.has_nface_elements(last):
            return OK

    celldim = PT.Zone.CellDimension(last)
    native_elts = PT.Zone.get_ordered_elements_per_dim(last)[celldim]
    tot_elts = sum(PT.Element.Size(e) for e in native_elts)

    n_cell = PT.Zone.n_cell(last)

    if n_cell != tot_elts:
        return f"Number of native elements is not equal to the number of cells (expected {n_cell}, found {tot_elts})"
    else:
        return OK

def zone_contiguous_elt_range(nodes:List[CGNSTree]) -> str: 
    """E236 - ElementRange contiguous numbering
    
    The ElementRange of all the Elements_t nodes must not overlap, and
    their combination must be contiguous.
    
    Erroneous tree examples:

    Zone Zone_t I4 [[876 1633 0]]
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [1 1633]
    └───BAR Elements_t I4 [3 0]  
        └───ElementRange IndexRange_t I4 \033[91m[1 116] #Overlaps [1 1633]\033[0m 

    Zone Zone_t I4 [[876 1633 0]]
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [1 1633]
    └───BAR Elements_t I4 [3 0]  
        └───ElementRange IndexRange_t I4 \033[91m[1701 1816] #Discontinuous \033[0m 

    Correct example:

    Zone Zone_t I4 [[876 1633 0]]
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [1 1633]
    └───BAR Elements_t I4 [3 0]  
        └───ElementRange IndexRange_t I4 \033[32m[1634 1750] #Correct \033[0m 
    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK

    elts = PT.Zone.get_ordered_elements(last)
    ranges = [PT.Element.Range(e) for e in elts]
    if len(ranges) > 0:
        if ranges[0][0] != 1:
            return f"Elements_t numbering should start at 1, but lowest ElementRange is " \
                   f"{ranges[0]} for node '{elts[0][0]}'"
        for i in range(len(elts)-1):
            if ranges[i+1][0] != (ranges[i][1]+1):
                return f"Elements_t numbering is not contiguous: range of '{elts[i+1][0]}'" \
                       f" does not follows range of '{elts[i][0]}'"

    return OK

def zone_ordered_elt_range(nodes:List[CGNSTree]) -> str: 
    """W237 - ElementRange ordering
    
    The Elements_t nodes must be ordered (in sense of their ElementRange)
    accordingly to their dimension (either increasing or decreasing).
    This is not stricly required by the CGNS standard, but several
    functionnalities of maia relie on this assumption.
    
    Erroneous tree example:

    Zone Zone_t I4 [[1000 2187 0]]
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [   1  486] \033[91m#Surfacic\033[0m
    ├───PYRA Elements_t I4 [12 0]
    │   └───ElementRange IndexRange_t I4 [ 487 2673] \033[91m#Volumic\033[0m
    └───QUAD Elements_t I4 [7 0]
        └───ElementRange IndexRange_t I4 [2674 2916] \033[91m#Surfacic\033[0m

    Correct example:

    Zone Zone_t I4 [[1000 2187 0]]
    ├───PYRA Elements_t I4 [12 0]
    │   └───ElementRange IndexRange_t I4 [   1 2187] \033[32m#Volumic\033[0m
    ├───TRI Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [2188 2673] \033[32m#Surfacic\033[0m
    └───QUAD Elements_t I4 [ 7 0]
        └───ElementRange IndexRange_t I4 [2674 2916] \033[32m#Surfacic\033[0m

    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK

    #Skip if zone has mixed elements (not relevant)
    zone_elts = set(PT.Element.Type(e) for e in PT.iter_children_from_label(last, 'Elements_t'))
    if 'MIXED' in zone_elts:
        return OK

    if PT.Zone.elt_ordering_by_dim(last) == 0:
        return 'Element sections are not ordered according to their dimension'
    else:
        return OK

def element_connectivity_size(nodes:List[CGNSTree]) -> str: 
    """E238 - Inconsistent element connectivity shape

    For Elements_t nodes, the size of ElementConnectivity, ElementStartOffset
    and ParentElements (if any) arrays must be consistent with the
    number of elements registered in the ElementRange node.

    Erroneous tree example:

    TRI Elements_t I4 [5 0]
    ├───ElementRange IndexRange_t I4 [1 10] \033[32m     # 10 tri defined\033[0m
    └───ElementConnectivity DataArray_t I4 \033[91m(33,) # Size should be 10*3=30\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) != 'Elements_t':
        return OK
    
    n_elem = PT.Element.Size(last)
    if PT.Element.Type(last) in ['NFACE_n', 'NGON_n', 'MIXED']:
        eso = PT.get_child_from_name(last, 'ElementStartOffset')
        if eso is not None: # No check can be done with CGNS3 version
            if (value:=PT.get_value(eso, raw=True)) is None:
                return f"Missing value for ElementStartOffset array"
            elif value.shape != (n_elem+1,):
                return f"Wrong size for ElementStartOffset array : expected {(n_elem+1,)}, got {value.shape}"
            # To check EC size, we need to get last value of ESO array
            # TODO : with better charging rule, if / else is useless
            expt_size = (value.last_value,) if hasattr(value, 'last_value') else (value[-1],)
            ec = PT.find_child_from_name(last, 'ElementConnectivity')
            if (ec_value:=PT.get_value(ec, raw=True)) is None:
                return f"Missing value for ElementConnectivity array"
            elif ec_value.shape != expt_size:
                return f"Wrong size for ElementConnectivity array : expected {expt_size}, got {ec_value.shape}"
    else:
        n_vtx = PT.Element.NVtx(last)
        ec = PT.find_child_from_name(last, 'ElementConnectivity')
        if (value:=PT.get_value(ec, raw=True)) is None:
            return f"Missing value for ElementConnectivity array"
        else:
            expt = (n_vtx*n_elem,)
            if value.shape != expt:
                return f"Wrong size for ElementConnectivity array : expected {expt}, got {value.shape}"

    # These nodes are not mandatory
    for name in ['ParentElements', 'ParentElementsPosition']:
        node = PT.get_child_from_name(last, name)
        if node is not None:
            if (value:=PT.get_value(node, raw=True)) is None:
                return f"Missing value for {name} array"
            elif value.shape != (n_elem,2):
                return f"Wrong size for {name} array : expected {(n_elem,2)}, got {value.shape}"

    return OK


def zone_coords_size(nodes:List[CGNSTree]) -> str: 
    """E239 - Zone coordinates array size
    
    Within a given Zone_t node, coordinates arrays size must be equal to the number of
    vertices of the zone.
    
    Erroneous tree example:

    Stator Zone_t I4 [[\033[32m72254\033[0m 60288  0]]
    └───GridCoordinates GridCoordinates_t
        ├───CoordinateX DataArray_t R8 (\033[32m72254\033[0m,)
        ├───CoordinateY DataArray_t R8 (\033[32m72254\033[0m,)
        └───CoordinateZ DataArray_t R8 (\033[91m72257\033[0m,)
    """
    if len(nodes) < 2 or PT.get_label(nodes[-2]) != 'GridCoordinates_t':
        return OK
    if PT.get_label(nodes[-1]) != 'DataArray_t' or PT.get_name(nodes[-1]) == 'CoordinateTransform':
        return OK

    zone = nodes[2]
    co = nodes[-1]

    n_vtx = PT.Zone.VertexSize(zone)

    if (nval := co[1]) is not None and (nshape := nval.shape) != n_vtx:
        return f"Shape of coordinate array is not consistent with zone's number of vertices:" \
               f" expected {n_vtx}, got {nshape}"
    else:
        return OK



def invalid_gridlocation_value(nodes:List[CGNSTree]) -> str:
    """E241 - Unexpected value for GridLocation node

    Depending of the kind and CellDimension of the parent zone,
    GridLocation_t values are restricted to:

    - Structured zones
        + CellDim=3 : CellCenter, {I|J|K}FaceCenter, {I|J|K}EdgeCenter, Vertex
        + CellDim=2 : CellCenter, {I|J}EdgeCenter, Vertex
        + CellDim=1 : CellCenter, Vertex
    - Unstructured zones
        + CellDim=3 : CellCenter, FaceCenter, EdgeCenter, Vertex
        + CellDim=2 : CellCenter, EdgeCenter, Vertex
        + CellDim=1 : CellCenter, Vertex

    Erroneous tree example:

    Stator Zone_t I4 [[72254 60288  0]]
    ├───ZoneType ZoneType_t \033[32m"Unstructured"\033[0m
    └───ZoneBC ZoneBC_t
        ├───bc_moyeu.4 BC_t "FamilySpecified"
        │   ├───GridLocation GridLocation_t \033[91m"IFaceCenter" # Not for unstructured zones\033[0m
        │   └───PointList IndexArray_t I4 (1, 864)
        └───bc_carter.5 BC_t "FamilySpecified"
            ├───GridLocation GridLocation_t \033[32m"FaceCenter"\033[0m
            └───PointList IndexArray_t I4 (1, 1632)
    """
    last = nodes[-1]
    if PT.get_label(last) == 'GridLocation_t':
        base = nodes[1]
        zone = nodes[2]
        loc = PT.get_str_value(last)
        if loc in ['EdgeCenter', 'FaceCenter'] and PT.Zone.Type(zone) == 'Structured':
            suff = '|'.join('IJK'[:PT.Zone.IndexDimension(zone)])
            return f"{loc} value can not be used for GridLocation on a structured zone, use {{{suff}}}{loc}"
        if loc[0] in 'IJK' and loc[1:] in ['EdgeCenter', 'FaceCenter'] and PT.Zone.Type(zone) == 'Unstructured':
            return f"{loc} value can not be used for GridLocation on an unstructured zone, use {loc[1:]}"
        if ('FaceCenter' in loc or loc[0] == 'K') and (celldim:=PT.get_np_value(base)[0]) < 3:
            return f"{loc} value can not be used for GridLocation on a CellDim={celldim} zone"
        if 'EdgeCenter' in loc and (celldim:=PT.get_np_value(base)[0]) < 2:
            return f"{loc} value can not be used for GridLocation on a CellDim={celldim} zone"
    return OK

def pointrange_normal_axis(nodes:List[CGNSTree]) -> str:
    """W242 - Structured subset GridLocation

    When a structured subset has {I|J|K}{Face|Edge}Center
    location, the related PointRange should have a consistent
    constant axis.
    This rule is warning because the difference can be desired,
    especially if internal edges or faces are described.

    Erroneous tree example:

    bc_7_005 BC_t "FamilySpecified"
    ├───GridLocation GridLocation_t "Vertex"
    ├───PointRange IndexRange_t I4 [[1 21] [1 1] [1 141]]
    └───BCDataSet#Init BCDataSet_t "Null"
        ├───GridLocation GridLocation_t \033[93m"IFaceCenter"\033[0m
        └───PointRange IndexRange_t I4 \033[93m[[1 20] [1 1] [1 140]] #Seems to be JFaceCenter\033[0m

    """
    last = nodes[-1]
    grid_loc = PT.get_child_from_name(last, 'GridLocation')
    pr = PT.get_child_from_name(last, 'PointRange')
    if pr is not None and grid_loc is not None and PT.get_str_value(grid_loc)[0] in 'IJK':
        pr_val = PT.get_np_value(pr)
        loc_val = PT.get_str_value(grid_loc)
        cst_axis = (pr_val[:,0] == pr_val[:,1]).tolist()
        if sum(cst_axis) == 1:
            axis = cst_axis.index(True)
            if loc_val[0] != 'IJK'[axis]:
                return f"GridLocation value is {loc_val}, but PointRange {pr_val.tolist()}" \
                       f" seems to be {'IJK'[axis]}{loc_val[1:]}"
            
    return OK

def missing_familyname(nodes:List[CGNSTree]) -> str: 
    """E243 - Missing FamilyName node

    If a BC_t node is of type "FamilySpecified", the FamilyName node is
    mandatory.

    Erroneous tree example:

    bc_142 BC_t \033[32m"FamilySpecified"\033[0m
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───PointList IndexArray_t I4 (1, 15168)
    └╴╴╴\033[91mMissing FamilyName node\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) == 'BC_t' and PT.get_str_value(last) == 'FamilySpecified':
        if PT.get_child_from_name(last, 'FamilyName') is None:
            return f"Missing FamilyName child for FamilySpecified BC"
    return OK

def unstructured_gc1to1(nodes:List[CGNSTree]) -> str: 
    """E246 - Unallowed GridConnectivity1to1_t

    GridConnectivity1to1_t nodes can only be used for structured zones.

    Erroneous tree example:

    ATB91 CGNSBase_t I4 [3 3]
    └───zone Zone_t I4 [[11193 7200 0]]
        ├───ZoneType ZoneType_t \033[32m"Unstructured"\033[0m
        └───ZoneGridConnectivity ZoneGridConnectivity_t
            ├───matchA \033[32mGridConnectivity_t\033[0m "zone"
            └───matchB \033[91mGridConnectivity1to1_t\033[0m "zone"
    """
    last = nodes[-1]
    if PT.get_label(last) == 'GridConnectivity1to1_t':
        if PT.Zone.Type(nodes[2]) != 'Structured':
            return f"GridConnectivity1to1_t nodes can only be used for Structured zones"
    return OK


def gc_transform_value(nodes:List[CGNSTree]) -> str: 
    """E247 - Transform value of GridConnectivity1to1_t

    For GridConnectivity1to1_t nodes, the value of the Transform child
    must be a permutation of ±[1,2,3] (or ±[1,2] for 2D grids).

    Erroneous tree example:

    CME2 CGNSBase_t I4 [3 3]
    ├───row_1_down Zone_t I4 [[21 20 0] [85 84 0] [141 140 0]]
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t
    │       └───gc_2_005 GridConnectivity1to1_t "row_1_inlet"
    │           └───Transform "int[IndexDimension]" I4 \033[91m[-1 2 -2]\033[0m
    └───row_1_inlet Zone_t I4 [[17 16 0] [85 84 0] [17 16 0]]
    """
    if len(nodes) < 2:
        return OK
    last = nodes[-1]
    parent = nodes[-2]
    if PT.get_name(last) == 'Transform' and PT.get_label(parent) == 'GridConnectivity1to1_t':
        zone = nodes[2]
        tr_val = PT.get_np_value(last)
        idx_dim = PT.Zone.IndexDimension(zone)
        if tr_val.size != idx_dim:
            return f"Invalid dimension: expected {idx_dim}, got {tr_val.size}"
        expt = list(range(1,idx_dim+1))
        if sorted(abs(k) for k in tr_val) != expt:
            return f"Transform value {tr_val.tolist()} is not a permutation of ±{expt}"
    return OK

def gc_transform_relation(nodes:List[CGNSTree]) -> str: 
    """E248 - Invalid Transform specification

    For GridConnectivity1to1_t nodes, the transformation matrix
    allows to compute the position of a point Index2 (in opposite
    zone) from a point Index1 (in current zone) following:

    Index2 = T.(Index1 - Begin1) + Begin2

    where Begin1 and Begin2 are respectively the first column of
    PointRange and PointRangeDonor children.

    If this relation is not satisfied for Index1 = End1
    (the second column of PointRange), the relation between
    Transform, PointRange and PointRangeDonor is invalid.

    Erroneous tree example:

    gc_2_005 GridConnectivity1to1_t "row_1_inlet"
    ├───Transform "int[IndexDimension]" I4 [-1 2 -3]
    ├───PointRange IndexRange_t I4 [[1 1] [1 85] [1 17]]
    └───PointRangeDonor IndexRange_t I4 [[1 1] [1 85] [1 17]]
    \033[91m# Relation gives Index2 = [1 85 -15] which is != PointRangeDonor[:,1]\033[0m

    Correct tree example:

    gc_2_005 GridConnectivity1to1_t "row_1_inlet"
    ├───Transform "int[IndexDimension]" I4 [-1 2 -3]
    ├───PointRange IndexRange_t I4 [[1 1] [1 85] [1 17]]
    └───PointRangeDonor IndexRange_t I4 [[1 1] [1 85] [17 1]]
    \033[32m# Relation gives Index2 = [1 85 1] which is == PointRangeDonor[:,1]\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) != 'GridConnectivity1to1_t':
        return OK

    pr = PT.get_np_value(PT.find_child_from_name(last, 'PointRange'))
    prd = PT.get_np_value(PT.find_child_from_name(last, 'PointRangeDonor'))
    if not (prd[:,1] == PT.utils.gc_transform_point(last, pr[:,1])).all():
        return f"Invalid relation between Transform, PointRange and PointRangeDonor"
    return OK




def field_shape(nodes:List[CGNSTree]) -> str:
    """E251 - Data field shape

    Under fields containers (FlowSolution, ZoneSubRegion, BCDataSet, ...),
    the shape of DataArray must be consistent with the number of elements
    of the related PointList/PointRange if any, or with zone shape
    otherwise (for FlowSolution_t/DiscreteData_t).

    Erroneous tree examples:

    Stator Zone_t I4 [[72254 60288  0]]
    └───ZSR ZoneSubRegion_t
        ├───GridLocation GridLocation_t "CellCenter"
        ├───PointList IndexArray_t I4 \033[32m(1, 894)\033[0m
        └───Pressure DataArray_t R8 \033[91m(800,) # Shape should be (894,)\033[0m

    row_1_down Zone_t I4 [[\033[32m21\033[0m 20 0] [\033[32m85\033[0m 84 0] [\033[32m141\033[0m 140 0]]
    └───FlowSolution#Init FlowSolution_t
        ├───GridLocation GridLocation_t "Vertex"
        └───Density DataArray_t R8 \033[91m(251685,) #Shape should be (21,85,141)\033[0m
    """
    last = nodes[-1]
    if PT.get_label(last) != 'DataArray_t':
        return OK

    parent = nodes[-2]
    parent_label = PT.get_label(parent)
    expt_shape = None
    if parent_label in ["FlowSolution_t", "DiscreteData_t"]:
        if PT.Container._is_subset(parent):
            expt_shape = (PT.Subset.n_elem(parent),)
        else:
            assert (loc:=PT.Container.GridLocation(parent)) in ['CellCenter', 'Vertex']
            zone = nodes[2]
            expt_shape = PT.Zone.CellSize(zone) if loc == 'CellCenter' else PT.Zone.VertexSize(zone)
    elif parent_label == 'ZoneSubRegion_t':
        expt_shape = (PT.Subset.n_elem(PT.Container.SubsetNode(parent, nodes[2])),)
    elif parent_label == 'BCData_t':
        pparent = nodes[-3]
        if PT.get_label(pparent) == 'BCDataSet_t':
            expt_shape = (PT.Subset.n_elem(PT.Container.SubsetNode(pparent, nodes[2])),)
        else: #FamilyBCDataSet_t
            expt_shape = (1,)

    if expt_shape is not None:
        val = PT.get_value(last, True)
        if val is None:
            return f"Missing value for DataArray_t node: expected shape is {expt_shape}"
        # NB : array of shape (1,) are allowed under BCData_t nodes
        elif (shape:=val.shape) != expt_shape and not (parent_label == 'BCData_t' and shape==(1,)):
            return f"Invalid shape for DataArray_t node: expected {expt_shape}, got {val.shape}"
    
    return OK

def unexpected_field_component(nodes:List[CGNSTree]) -> str:
    """W252 - Unexpected tensorial field component

    The number of fields used to describe a vectorial or tensorial
    cartesiant field must be consistent with the physical dimension
    of the mesh.

    Erroneous tree examples:

    Base CGNSBase_t I4 [2 \033[32m2\033[0m]
    └───zone Zone_t I4 [[876 1633  0]]
        └───FlowSolution@Vertex@Init FlowSolution_t
            ├───Density IndexArray_t R8 (876,)
            ├───\033[32mMomentumX\033[0m IndexArray_t R8 (876,)
            ├───\033[32mMomentumY\033[0m IndexArray_t R8 (876,)
            └───\033[93mMomentumZ\033[0m IndexArray_t R8 (876,) \033[93m# Unexpected because phydim=2\033[0m
    """
    # Only cartesian for now
    containers = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCData_t']
    last = nodes[-1]
    if PT.get_label(last) == 'DataArray_t' and PT.get_label(nodes[-2]) in containers:
        phydim = PT.get_np_value(nodes[1])[1]
        forbidden = set()
        if phydim < 3:
            forbidden |= {'Z', 'XZ', 'YZ', 'ZX', 'ZY', 'ZZ'}
        if phydim < 2:
            forbidden |= {'Y', 'XY', 'YX', 'YY'}
        for suff in forbidden:
            if PT.get_name(last).endswith(suff):
                return f"Component {suff} of tensorial field is unexepected since PhysicalDimension of mesh is {phydim}"
            
    return OK

def vectorial_field_component(nodes:List[CGNSTree]) -> str:
    """W253 - Missing tensorial field component

    The number of fields used to describe a vectorial or tensorial
    cartesiant field must be consistent with the physical dimension
    of the mesh.

    Erroneous tree examples:

    Base CGNSBase_t I4 [3 \033[32m3\033[0m]
    └───zone Zone_t I4 [[876 1633  0]]
        └───FlowSolution@Vertex@Init FlowSolution_t
            ├───Density IndexArray_t R8 (876,)
            ├───\033[32mMomentumX\033[0m IndexArray_t R8 (876,)
            ├───\033[32mMomentumY\033[0m IndexArray_t R8 (876,)
            ╵╴╴╴\033[93mMissing MomentumZ component (because phydim=3)\033[0m
    """
    # Only cartesian for now
    containers = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCData_t']
    last = nodes[-1]
    if PT.get_label(last) == 'DataArray_t' and PT.get_label(nodes[-2]) in containers:
        phydim = PT.get_np_value(nodes[1])[1]
        field = PT.get_name(last)
        tensor = set()
        vector = set()
        if phydim >= 2:
            tensor |= {'XX', 'XY', 'YX', 'YY'}
            vector |= {'X', 'Y'}
        if phydim >= 3:
            tensor |= {'XZ', 'YZ', 'ZX', 'ZY', 'ZZ'}
            vector |= {'Z'}
        # Maybe we should check at container level to raise the error only once,
        # but more difficult to report all fields at once
        if any(field.endswith(suff) for suff in tensor):
            missing = {s for s in tensor if PT.get_child_from_name(nodes[-2], field[:-2]+s) is None}
            if len(missing) > 0:
                maybe_symetric = missing == {'YX', 'ZX', 'ZY'} if phydim == 3 else missing == {'YX'}
                if maybe_symetric:
                    return f"Missing lower component(s) for tensorial field {field[:-2]}"\
                           f" (this may be intentional if field is symmetric)"
                else:
                    return f"Missing {missing} component(s) for tensorial field {field[:-2]}"
        elif any(field.endswith(suff) for suff in vector):
            missing = [v for v in vector if PT.get_child_from_name(nodes[-2], field[:-1]+v) is None]
            if len(missing) > 0:
                return f"Missing {missing} component(s) for vectorial field {field[:-1]}"
            
    return OK

def close_to_dataname_identifier(nodes:List[CGNSTree]) -> str:
    """W254 - Close to dataname identifier

    Data fields should be named from conventional identifiers:
    https://cgns.org/standard/SIDS/convention.html

    This rule warn the user if the name of the field is close
    to a known identifier.

    Erroneous tree examples:

    FlowSolution@Vertex@Init FlowSolution_t
    ├───\033[32mDensity\033[0m IndexArray_t R8 (876,)
    ├───\033[93mTemprature\033[0m IndexArray_t R8 (876,) \033[93m# Did you mean Temperature ?\033[0m
    └───\033[32mPressure\033[0m IndexArray_t R8 (876,)
    """
    # 2 rules so we can disable the second one
    containers = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCData_t']
    last = nodes[-1]
    if PT.get_label(last) == 'DataArray_t' and PT.get_label(nodes[-2]) in containers:
        field = PT.get_name(last)
        if not field in DATANAME_IDENTIFIERS:
            closest = difflib_cache[field]
            if len(closest) > 0:
                return f"Unconventional field name, did you mean '{closest[0]}' ?"
            
    return OK

def unconventional_identifier(nodes:List[CGNSTree]) -> str:
    """W255 - Not a conventional identifier

    Data fields should be named from conventional identifiers:
    https://cgns.org/standard/SIDS/convention.html

    This rule warn the user if the name of the field is not
    a known identifier.

    Erroneous tree examples:

    FlowSolution@Vertex@Init FlowSolution_t
    ├───\033[32mDensity\033[0m IndexArray_t R8 (876,)
    ├───\033[93mextrp_on(temp)\033[0m IndexArray_t R8 (876,) \033[93m# Unknow dataname identifier\033[0m
    └───\033[32mPressure\033[0m IndexArray_t R8 (876,)
    """
    # 1 rule or two rules ? Maybe two so we can disable the second one
    containers = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCData_t']
    last = nodes[-1]
    if PT.get_label(last) == 'DataArray_t' and PT.get_label(nodes[-2]) in containers:
        field = PT.get_name(last)
        if not field in DATANAME_IDENTIFIERS:
            closest = difflib_cache[field]
            if len(closest) == 0:
                return f"Unconventional field name"
            
    return OK





_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)

NODE_RULES = {func[1].__doc__[:4] : func[1] for func in _funcs}
assert len(NODE_RULES) == len(_funcs)