import sys
import inspect

import maia.pytree      as PT
import maia.pytree.pred as PTp

from maia.pytree.typing import List, CGNSTree # Strangly import * brings NamedTuple as a function

from maia.pytree.cgns_keywords import dtype_to_cgns

from .data import LABEL_PROPS

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
    if PT.get_label(last) != 'Zone_t':
        return OK

    celldim = PT.Zone.CellDimension(last)
    native_elts = PT.Zone.get_ordered_elements_per_dim(last)[celldim]
    tot_elts = sum(PT.Element.Size(e) for e in native_elts)

    n_cell = PT.Zone.n_cell(last)

    if n_cell != tot_elts:
        return f"Number of native elements is not equal to the number of cells (expected {n_cell}, found {tot_elts})"
    else:
        return OK

def unexpected_child_label(nodes:List[CGNSTree]) -> str: 
    """E209 - Unexpected child label
    
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