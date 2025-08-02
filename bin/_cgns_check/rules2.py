import sys
import inspect
import collections
import h5py

import maia.pytree      as PT
import maia.pytree.pred as PTp

from maia.pytree.typing import List, CGNSTree # Strangly import * brings NamedTuple as a function

from maia.pytree.cgns_keywords import dtype_to_cgns

from .data import LABEL_PROPS, ALL_LABELS

OK = ''
class Colors:
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def sibling_zones_inttype_consistency(nodes:List[CGNSTree]) -> str: 
    """E202 - Sibling zones integer type consistency
    
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
    """E203 - Zone integer type consistency
    
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

    if PT.get_label(last) != 'Zone_t':
        return OK

    ztype = PT.get_np_value(last).dtype
    wrong = None

    names = ['ElementRange', 'ElementConnectivity', 'ElementStartOffset', 'ParentElements']
    for elt, node in PT.iter_children_from_predicates(last, ['Elements_t', PTp.name_in(names)], ancestors=True):
        if (nval := node[1]) is not None and (ntype := nval.dtype) != ztype:
            wrong = (f'{elt[0]}/{node[0]}', ntype)
            break

    if wrong is None:
        labels = ['IndexRange_t', 'IndexArray_t']
        for sub in PT.iter_all_subsets(last):
            for node in PT.iter_children_from_predicate(sub, PT.pred.label_in(labels)):
                if (nval := node[1]) is not None and (ntype := nval.dtype) != ztype:
                    wrong = (f'{sub[0]}/{node[0]}', ntype)
                break

    if wrong is not None:
        wpath, wtype = wrong
        return f"Integer type of zone is {dtype_to_cgns[ztype]}," \
               f" but some connectivity arrays are of kind {dtype_to_cgns[wtype]} (eg. {wpath})"
    else:
        return OK


def zone_coords_size(nodes:List[CGNSTree]) -> str: 
    """E204 - Zone coordinates array size
    
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

    zone = nodes[2]
    co = nodes[-1]

    n_vtx = PT.Zone.VertexSize(zone)

    if (nval := co[1]) is not None and (nshape := nval.shape) != n_vtx:
        return f"Shape of coordinate array is not consistent with zone's number of vertices:" \
               f" expected {n_vtx}, got {nshape}"
    else:
        return OK

def zone_physical_dimension(nodes:List[CGNSTree]) -> str: 
    """E205 - Zone physical dimension
    
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

def zone_cell_dimension(nodes:List[CGNSTree]) -> str: 
    """E206 - Zone cell dimension
    
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
    
    if (zdim := PT.Zone.CellDimension(last)) != cell_dim:
        return f"Number of coordinates array is not consistent with physical dimension of the parent base:" \
               f" expected {cell_dim}, got {zdim}"

    return OK

def zone_contiguous_elt_range(nodes:List[CGNSTree]) -> str: 
    """E207 - ElementRange contiguous numbering
    
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

def zone_number_of_elements(nodes:List[CGNSTree]) -> str: 
    """E208 - Number of native mesh elements
    
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

    celldim = PT.Zone.CellDimension(last)
    native_elts = PT.Zone.get_ordered_elements_per_dim(last)[celldim]
    tot_elts = sum(PT.Element.Size(e) for e in native_elts)

    n_cell = PT.Zone.n_cell(last)

    if n_cell != tot_elts:
        return f"Number of native elements is not equal to the number of cells (expected {n_cell}, found {tot_elts})"
    else:
        return OK

def invalid_label(nodes:List[CGNSTree]) -> str: 
    """E209 - Invalid label
    
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

def unexpected_child_label(nodes:List[CGNSTree]) -> str: 
    """E210 - Unexpected child label
    
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
    """E213 - Use of reserved name
    
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
    """E214 - Predefined name
    
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

def duplicated_children_name(nodes:List[CGNSTree]) -> str:
    """E215 - Duplicated children name

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

def exclusive_children(nodes:List[CGNSTree]) -> str:
    """E216 - Mutually exclusive children

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

def S_zone_size(nodes:List[CGNSTree]) -> str:
    """E217 - Size of structured zone

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

def too_many_children(nodes:List[CGNSTree]) -> str:
    """E211 - Too many children of same label
    
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
    """E212 - Missing children of specific label
    
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


def zone_ordered_elt_range(nodes:List[CGNSTree]) -> str: 
    """W201 - ElementRange ordering
    
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
    if PT.Zone.elt_ordering_by_dim(last) == 0:
        return 'Element sections are not ordered according to their dimension'
    else:
        return OK

def missing_family(nodes:List[CGNSTree]) -> str: 
    """E251 - Missing Family_t node

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
    """E252 - Missing FamilyBC_t node

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
    
def missing_familyname(nodes:List[CGNSTree]) -> str: 
    """E253 - Missing FamilyName node

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


def element_connectivity_size(nodes:List[CGNSTree]) -> str: 
    """E254 - Inconsistent element connectivity shape

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

def unstructured_gc1to1(nodes:List[CGNSTree]) -> str: 
    """E261 - Unallowed GridConnectivity1to1_t

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

def gc_opposite_zone(nodes:List[CGNSTree]) -> str: 
    """E262 - Missing opposite zone

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

def gc_transform_value(nodes:List[CGNSTree]) -> str: 
    """E263 - Transform value of GridConnectivity1to1_t

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
    """E264 - Invalid Transform specification

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

# Pour chaque noeud, on veut représetner: 
# - une liste de labels autorisés
# - le nombre d'elts associé à chaque label (1, N ou open bar)
# - des exclusions (si A présent, B interdit)
# - Des valeurs imposées (pour les enums) 
# - si le nom des enfants est imposé ou freestyle
#
#
#
#



class CGNSRule:
    def __init__(self, name, code, doc, check) -> None:
        self.name = name
        self.code = code
        self.doc = doc
        self.check_fn = check

        self.out = ''

    def check(self, nodes:List[CGNSTree]) -> bool:
        self.out = self.check_fn(nodes)
        return self.out == OK



STAGE_2_RULES_OLD = [
    CGNSRule(
        'Zone integer consistency',
        'E202',
        inspect.cleandoc(sibling_zones_inttype_consistency.__doc__),
        sibling_zones_inttype_consistency,
    ),
    CGNSRule(
        'Zone integer consistency',
        'E203',
        inspect.cleandoc(zone_inttype_consistency.__doc__),
        zone_inttype_consistency
    )
]



_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)

NODE_RULES = {func[1].__doc__[:4] : func[1] for func in _funcs}
assert len(NODE_RULES) == len(_funcs)