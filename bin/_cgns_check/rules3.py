import numpy as np
import sys
import inspect
from mpi4py      import MPI
from collections import defaultdict

import maia
import maia.pytree      as PT
import maia.pytree.pred as PTp
import maia.pytree.maia as MT

from maia.transfer import protocols as EP

from maia.utils import np_utils, par_utils
from maia.utils import vstride as vs

from maia.typing import List, CGNSTree, MPIComm # Strangly import * brings NamedTuple as a function



OK = ''


IS_POLY = PTp.IS_POLY2D_ZONE | PTp.IS_POLY3D_ZONE
IS_GC_MATCH = PT.pred.is_gc_of_kind(is_1to1=True)

class utils:
    # Just a namespace to store utils functions

    @staticmethod
    def defaultdict_sum(d1, d2):
        return defaultdict(int, {k: d1[k]+d2[k] for k in set(d1) | set(d2)})

    @staticmethod
    def dist_pl_value(subset):
        if (pl := PT.get_child_from_name(subset, 'PointList')) is not None:
            return PT.get_np_value(pl)[0]
        elif (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
            distri = MT.distribution_value(subset, 'Index')
            return np_utils.single_dim_pr_to_pl(PT.get_np_value(pr), distri)[0]
        raise RuntimeError

    @staticmethod
    def get_zone_elements(zone:CGNSTree) -> List[CGNSTree]:
        """ Return the elements of a Zone_t node, but add if missing implicit
        NFace/NGon node (with juste ElementRange) """
        elts = PT.get_children_from_label(zone, 'Elements_t')
        if PTp.IS_POLY2D_ZONE(zone) and not PT.Zone.has_ngon_elements(zone):
            last = PT.Element.Range(MT.Zone.EdgeNode(zone))[1]
            erange = np.array([last+1, last+PT.Zone.n_cell(zone)], last.dtype)
            elts.append(PT.new_NGonElements(erange=erange))
        elif PTp.IS_POLY3D_ZONE(zone) and not PT.Zone.has_nface_elements(zone):
            last = PT.Element.Range(PT.Zone.NGonNode(zone))[1]
            erange = np.array([last+1, last+PT.Zone.n_cell(zone)], last.dtype)
            elts.append(PT.new_NFaceElements(erange=erange))
        return elts

    @staticmethod
    def compute_distri(tab, comm):
        gmax = comm.allreduce(tab.max(), MPI.MAX)
        gmin = comm.allreduce(tab.min(), MPI.MIN)
        length = int(gmax) - int(gmin) + 1
        div = length // comm.size
        rmd = length - div*comm.size
        distri = np.empty(comm.size+1, tab.dtype)
        distri[0] = gmin
        for i in range(comm.size):
            distri[i+1] = distri[i] + div + (i < rmd)
        
        return distri

class MatchingJnsTable:
    def __init__(self):
        self.computed = False
        self.table = defaultdict(list)
    def __getitem__(self, key):
        return self.table[key]
    def compute(self, tree, comm):
        from maia.algo.dist.matching_jns_tools import _create_local_match_table
        
        gc_paths = PT.predicates_to_paths(tree, ["CGNSBase_t", "Zone_t", "ZoneGridConnectivity_t", IS_GC_MATCH])

        local_match_table = _create_local_match_table(tree,
                                                      [PT.get_node_from_path(tree, p) for p in gc_paths],
                                                      [PT.utils.path_head(p, 2) for p in gc_paths],
                                                      comm)

        global_match_table = np.empty_like(local_match_table)
        comm.Allreduce(local_match_table, global_match_table, op=MPI.LAND)

        rows, cols = np.where(global_match_table)

        for r, c in zip(rows, cols):
            self.table[gc_paths[r]].append(gc_paths[c])

        self.computed = True

matching_jns_table = MatchingJnsTable()

# Errors code 300-399


def eso_values(nodes:List[CGNSTree], comm:MPIComm) -> str: 
    """E301 - ElementStartOffset values
    
    ElementStartOffset arrays should have the following properties:
        - first value is 0
        - array is stricly increasing (eso[i] < eso[i+1])
        - last value is the size of related ElementConnectivity array
    
    Erroneous tree example:

    NGonElements Elements_t I4 [22 0]
    ├───ElementRange IndexRange_t I4 [13 16]
    ├───ElementStartOffset DataArray_t I4 [0 4 \033[91m14\033[m 12 16]
    └───ElementConnectivity DataArray_t I4 (16,)
    """
    # Careful : we need an additional rule that report missing ESO on NGON/NFACE/MIXED TODO
    last = nodes[-1]
    if PT.get_name(last) == 'ElementStartOffset' and PT.get_label(nodes[-2]) == 'Elements_t':
        eso = PT.get_np_value(last)
        if not (st:=comm.bcast(eso[0], root=0)) == 0:
            return f"ESO array should start at 0, but starting value is {st}"
            
        mask = ~(eso[:-1] < eso[1:])
        lsum = mask.sum()
        if (gsum:=comm.allreduce(lsum)) > 0:
            distri = MT.distribution_value(nodes[-2], 'Element')
            lval = np.where(mask)[0][0] + distri[0] if lsum > 0 else distri[2]+1
            gval = comm.allreduce(lval, MPI.MIN)
            return f"ESO array is not strictly increasing (ESO[i] < ESO[i+1]) : {gsum} indices are wrong, first one beeing {gval}"

    return OK


def left_parent_for_bnd_cells(nodes:List[CGNSTree], comm:MPIComm) -> str: 
    """E302 - Right parent for boundary entities

    Faces (resp. edges) on the boundary of a 3D (resp. 2D) domain
    should have a second parent (pe[entiy,1]) set to zero.
    In other words, first parent (pe[entity,0]) should never be zero.

    Erroneous tree example:

    EdgeElements Elements_t I4 [3 0]
    ├───ElementRange IndexRange_t I4 [1 7]
    ├───ElementConnectivity DataArray_t I4 (14,)
    └───ParentElements DataArray_t I4 (7,2)
        [[8 0]]
        [[8 9]]
        [[9 0]]
        [[8 0]]
        \033[91m[[0 9]]\033[m
        [[8 0]]
        [[9 0]]
    """
    #TODO maybe warning ? pas sur que le cgns dise que second = 2e position
    last = nodes[-1]
    if PT.get_name(last) == 'ParentElements' and PT.get_label(nodes[-2]) == 'Elements_t':

        has_right_parent = PT.get_np_value(last)[:,0] == 0
        if (gsum:=comm.allreduce(has_right_parent.sum())) > 0:
            return f"Some lowerdim elements have no first parent: {gsum} elements are wrong"

    return OK

def zone_cell_dimension_mixed(nodes:List[CGNSTree], comm:MPIComm) -> str: 
    """E303 - Zone cell dimension (MIXED elements)

    The maximal dimension of elements a given Zone_t must be consistent with the cell
    dimension of its CGNSBase parent.
    This rule is similar to E231, but analyse content of MIXED elements to compute zone dimension.

    Erroneous tree example:

    Base CGNSBase_t I4 [\033[91m3\033[m 3]                       \033[91m# CellDim for base is 3\033[m
    └───zone Zone_t I4 [[4 2 0]]
        ├───ZoneType ZoneType_t "Unstructured"
        └───MixedElements Elements_t I4 [20  0]
            ├───ElementRange IndexRange_t I4 [1 6]
            ├───ElementConnectivity DataArray_t I4
            │   [5 1 2 3
            │    \033[91m5\033[m 4 3 2                           \033[91m# Highest dimension element is a TRI_3 (dim=2)\033[m
            │    3 1 2
            │    3 4 3
            │    3 3 1
            │    3 2 4]
            └───ElementStartOffset DataArray_t I4 [ 0  4  8 11 14 17 20]
    """
    from maia.pytree.sids import elements_utils as EU
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK
    if PT.get_child_from_predicate(last, PTp.is_element_of_type('MIXED')) is None:
        return OK # Case already treated in E231

    base = nodes[-2]
    assert PT.get_label(base) == 'CGNSBase_t'
    cell_dim = PT.get_np_value(base)[0]

    # This copy of zone has no mixed elements
    std_zone = PT.shallow_copy(last)
    PT.rm_children_from_predicate(std_zone, PTp.is_element_of_type('MIXED'))
    zone_dim = -1
    if PT.get_child_from_label(std_zone, 'Elements_t') is not None:
        zone_dim = PT.Zone.CellDimension(std_zone)

    # Here we deal MIXED elements
    all_types = set()
    for elt in PT.iter_children_from_predicate(last, PTp.is_element_of_type('MIXED')):
        elem_cnt = MT.Element.connectivity(elt)
        elem_eso_loc = elem_cnt.displs[:-1]
        elem_types   = elem_cnt.values[elem_eso_loc]

        local_types = set(np.unique(elem_types))
        all_types |= comm.allreduce(local_types, op = lambda s1,s2 : s1 | s2)

    for type in sorted(all_types):
        zone_dim = max(zone_dim, EU.id_to_dim(type))

    if zone_dim != cell_dim:
        return f"Maximal dimension of zone elements not consistent with cell dimension of the parent base:" \
               f" expected {cell_dim}, got {zone_dim}"

    return OK

def zone_number_of_elements_mixed(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E304 - Number of native mesh elements (MIXED elements)

    The total number of mesh elements whose dimension equals CellDimension
    must be equal to the number of elements of the Zone_t node.
    This rule is similar to E235, but analyse content of MIXED elements to compute number of elements

    Erroneous tree example:

    Base CGNSBase_t I4 [2 3]
    └───zone Zone_t I4 [[4 \033[32m3\033[m 0]]                   \033[32m# Zone should have three 2D cells\033[m
        ├───ZoneType ZoneType_t "Unstructured"
        └───MixedElements Elements_t I4 [20  0]
            ├───ElementRange IndexRange_t I4 [1 6]
            ├───ElementConnectivity DataArray_t I4
            │   [\033[91m5\033[m 1 2 3
            │    \033[91m5\033[m 4 3 2                           \033[91m# Only two 2D cells (TRI_3) are present\033[m
            │    3 1 2
            │    3 4 3
            │    3 3 1
            │    3 2 4]
            └───ElementStartOffset DataArray_t I4 [ 0  4  8 11 14 17 20]
    """
    from maia.pytree.sids import elements_utils as EU

    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK

    n_cell_zone = PT.Zone.n_cell(last)

    is_mixed = PT.get_child_from_predicate(last, PTp.is_element_of_type('MIXED')) is not None
    is_poly2d = PTp.IS_POLY2D_ZONE(last)
    is_poly3d = PTp.IS_POLY3D_ZONE(last)
    is_poly   = is_poly2d or is_poly3d
    if not (is_poly or is_mixed): # Std elements treated in E235
        return OK

    assert not (is_poly and is_mixed)

    cell_dim = PT.Base.CellDimension(nodes[-2])

    if is_mixed:
        all_elts = PT.get_nodes_from_label(last, 'Elements_t')
        std_elts = [elt for elt in all_elts if PT.Element.Type(elt) != 'MIXED']
        mix_elts = [elt for elt in all_elts if PT.Element.Type(elt) == 'MIXED']
        
        loc_cnt = defaultdict(int)
        for elt in mix_elts:
            elem_cnt = MT.Element.connectivity(elt)
            elem_eso_loc = elem_cnt.displs[:-1]
            elem_types   = elem_cnt.values[elem_eso_loc]
            kind, count = np.unique(elem_types, return_counts=True)
            for k, c in zip(kind, count):
                loc_cnt[k] += c
        cnt = comm.allreduce(loc_cnt, op=utils.defaultdict_sum)
        n_cell_mix = sum(val for key,val in cnt.items() if EU.id_to_dim(key)==cell_dim)
        n_cell_std = sum(PT.Element.Size(elt) for elt in std_elts if PT.Element.Dimension(elt) == cell_dim)

        n_cell = n_cell_mix + n_cell_std

    else: # is_poly
        has_native_elts = PT.Zone.has_ngon_elements(last) if is_poly2d else PT.Zone.has_nface_elements(last)
        if has_native_elts:
            return OK # Tested in E235

        node = MT.Zone.EdgeNode(last) if is_poly2d else PT.Zone.NGonNode(last)
        pe = PT.find_child_from_name(node, 'ParentElements')
        val = PT.get_np_value(pe).ravel()
        cell_ids = val[val != 0]
        min_id = comm.allreduce(np.min(cell_ids, initial=np.iinfo(cell_ids.dtype).max), MPI.MIN)
        max_id = comm.allreduce(np.max(cell_ids, initial=np.iinfo(cell_ids.dtype).min), MPI.MAX)
        n_cell = max_id - min_id + 1

    if n_cell != n_cell_zone:
        return f"Number of native elements is not equal to the number of cells (expected {n_cell_zone}, found {n_cell})"
    return OK


def unflagged_bc_elements(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E305 - Unflagged boundary elements

    For 3D (resp. 2D) polyedric meshes, boundary faces (resp. edges) should
    be present in at least one BC_t or GridConnectivity_t node.

    Erroneous tree example:

    zone Zone_t I4 [[12 2 0]]
    ├───ZoneType ZoneType_t "Unstructured"
    ├───NGonElements Elements_t I4 [22  0]
    │   ├───ElementRange IndexRange_t I4 [1 11]
    │   ├───ElementStartOffset DataArray_t I4 (12,)
    │   ├───ElementConnectivity DataArray_t I4 (44,)
    │   └───ParentElements DataArray_t I4 (11,2)
    │       [[12 0]]
    │       [[12 13]]
    │       [[13 0]]
    │       [[12 0]]
    │       [[13 0]]
    │       [[12 0]]
    │       [[13 0]]
    │       [[12 0]]
    │       [[13 0]]
    │       [[12 \033[32m0\033[m]] \033[91m# Face n°10 is a boundary face ...\033[m
    │       [[13 0]]
    └───ZoneBC ZoneBC_t
        └───BCs BC_t "BCFarfield"
            ├───GridLocation GridLocation_t "FaceCenter"
            └───PointList IndexArray_t I4 [[1 3 4 5 6 7 8 9 11]] \033[91m#... but does not appear in any BC_t or GC_t\033[m

    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t' or not IS_POLY(last):
        return OK

    # NG/Edge node is needed
    if PTp.IS_POLY2D_ZONE(last):
        maia.algo.ngon_to_edge_pe(last, comm)
        node = MT.Zone.EdgeNode(last)
        loc  = 'EdgeCenter'
    else:
        maia.algo.nface_to_pe(last, comm)
        node = PT.Zone.NGonNode(last)
        loc  = 'FaceCenter'

    distri = MT.distribution_value(node, 'Element')
    pe = PT.get_np_value(PT.find_child_from_name(node, 'ParentElements'))
    is_bnd = np.logical_or.reduce(pe==0, axis=1)

    ctn_pred = PTp.label_in(['ZoneBC_t', 'ZoneGridConnectivity_t'])
    pred = PTp.label_in(['BC_t', 'GridConnectivity_t']) & PTp.has_location(loc)
    pls = []
    for subset in PT.iter_children_from_predicates(last, [ctn_pred, pred]):
        pls.append(utils.dist_pl_value(subset))

    GI = EP.GlobalIndexer(distri, [pl-PT.Element.Range(node)[0] for pl in pls], comm)
    undefined = (GI.access_counts == 0) & is_bnd
    if (n_undf:=comm.allreduce(undefined.sum())) > 0:
        n_bnd = comm.allreduce(is_bnd.sum())
        entity = loc.replace('Center', '').lower() + 's'
        return f"Some boundary {entity} are not flagged in any subsets ({n_undf} found over {n_bnd} boundary {entity})"
    return OK

def multiflagged_bc_elements(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E306 - Redondant boundary elements

    For 3D (resp. 2D) polyedric meshes, boundary faces (resp. edges) should
    be present in at most one BC_t or GridConnectivity_t node.

    Erroneous tree example:

    zone Zone_t I4 [[12 2 0]]
    ├───ZoneType ZoneType_t "Unstructured"
    └───ZoneBC ZoneBC_t 
        ├───Inflow BC_t "UserDefined"
        │   ├───GridLocation GridLocation_t "FaceCenter"
        │   └───PointList IndexArray_t I4 [[\033[32m1\033[m]]
        ├───Outflow BC_t "UserDefined"
        │   ├───GridLocation GridLocation_t "FaceCenter"
        │   └───PointList IndexArray_t I4 [[\033[91m1\033[m 3]] \033[91m# Already present in 'Inflow'\033[m
        └───Sym BC_t "UserDefined"
            ├───GridLocation GridLocation_t "FaceCenter"
            └───PointList IndexArray_t I4 [[ 4  5  6  7  8  9 10 11]]

    """
    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t' or PT.Zone.Type(last) == 'Structured':
        return OK

    cell_dim = PT.Base.CellDimension(nodes[-2])
    locs = {1: ['CellCenter'],
            2: ['EdgeCenter', 'CellCenter'],
            3: ['FaceCenter', 'CellCenter']}[cell_dim]

    for loc in locs:
        ctn_pred = PTp.label_in(['ZoneBC_t', 'ZoneGridConnectivity_t'])
        pred = PTp.label_in(['BC_t', 'GridConnectivity_t']) & PTp.has_location(loc)
        pls = []
        names = []
        for subset in PT.iter_children_from_predicates(last, [ctn_pred, pred]):
            pls.append(utils.dist_pl_value(subset))
            names.append(PT.get_name(subset))

        distri = par_utils.distribution_from_gnum(pls, comm, full=True)
        GI = EP.GlobalIndexer(distri, [pl-1 for pl in pls], comm)
        if (dupl:=comm.allreduce((GI.access_counts > 1).sum())) > 0:
            # Get subset names
            send_ids = [(np.ones(pl.size, np.int32), i*np.ones(pl.size, np.int32)) for i,pl in enumerate(pls)]
            recv_ids = vs.from_counts(*GI.Put_v(send_ids, extend=True))
            dupl_ids = vs.take(recv_ids, np.flatnonzero(recv_ids.counts > 1)).values
            dupl_bcs = {names[i] for i in np.unique(dupl_ids)}

            dupl_bcs_g = comm.allreduce(dupl_bcs, op = lambda s1,s2 : s1 | s2)

            return f"Duplicated elements ids in {loc} boundary subsets ({dupl} found in subsets {dupl_bcs_g})"

    return OK


def pe_range_value(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E307 - Invalid ParentElements values

    The elements ids provided in face (resp. edge) ParentElements of a 3D
    (resp. 2D) mesh must be included in cell (resp. face) ElementRange.

    Erroneous tree example:

    zone Zone_t I4 [[9 8 0]]
    ├───ZoneType ZoneType_t "Unstructured"
    ├───EdgeElements Elements_t I4 [3  0]
    │   ├───ElementRange IndexRange_t I4 [1 16]
    │   ├───ElementConnectivity DataArray_t I4 (32,)
    │   └───ParentElements DataArray_t I4 (16, 2)
    │       [[\033[91m1\033[m 0]]
    │       [[\033[91m1\033[m 0]] \033[91m# These edges provides numbers \033[m 
    │       [[\033[91m3\033[m 0]] \033[91m# of parent faces in range [1,8]\033[m
    │       [[\033[91m1 2\033[m]]
    │        ...
    │       [[\033[91m8\033[m 0]]
    │       [[\033[91m8\033[m 0]]
    └───NGonElements Elements_t I4 [22 0]
        └───Element Range IndexRange_t I4 [\033[32m17 24\033[m] \033[91m# but range of face elements is [17,24]\033[m
            ├───ElementStartOffset DataArray_t I4 (9,)
            └───ElementConnectivity DataArray_t I4 (24,)
    """
    last = nodes[-1]
    if PT.get_name(last) == 'ParentElements' and PT.get_label(nodes[-2]) == 'Elements_t':
        val = PT.get_np_value(last).ravel()
        cell_ids = val[val != 0]
        is_ok = np.zeros(cell_ids.size, bool)

        zone = nodes[-3]
        target_dim = PT.Element.Dimension(nodes[-2]) + 1
        is_elt_of_dim = PTp.label_is('Elements_t') & PTp.NodePredicate(lambda n : PT.Element.Dimension(n) == target_dim)
        target_ranges = [PT.Element.Range(e) for e in PT.get_children_from_predicate(zone, is_elt_of_dim)]

        # For poly meshes, we allow implicit NFace (resp. NGon) definition
        if len(target_ranges) == 0 and IS_POLY(zone):
            cur_range = PT.Element.Range(nodes[-2])
            implicit_range = np.array([1, PT.Zone.n_cell(zone)]) + cur_range[1]
            target_ranges.append(implicit_range)

        for (start, end) in target_ranges:
            in_range = (start <= cell_ids) & (cell_ids <= end)
            is_ok |= in_range

        n_wrong_loc = is_ok.size - is_ok.sum()
        if (n_wrong:=comm.allreduce(n_wrong_loc)) > 0:
            return f"Some values of ParentElements does not refer to {target_dim}D elements ({n_wrong} are wrong)"

    return OK

def invalid_vtx_subset_id(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E308 - Out of range vertex subset values

    The values of vertex-located PointList must be included in range [1, n_vtx].

    Erroneous tree example:

    FlatPlate Zone_t I4 [[\033[32m876\033[m 1633 0]]
    ├───ZoneType ZoneType_t "Unstructured"
    └───ZoneSubRegion ZoneSubRegion_t
        ├───GridLocation GridLocation_t "Vertex"
        ├───PointList IndexArray_t I4 [[\033[32m400 600 \033[91m900\033[m]] \033[91m # Maximal vtx id is 876\033[m
        └───Density DataArray_t R8 [1.013 1.014 1.013]

    """
    last = nodes[-1]
    if PT.get_name(last) == 'PointList' and PT.Subset.GridLocation(nodes[-2]) == 'Vertex':
        # zone node can be at diffent level
        zone = next(node for node in nodes if PT.get_label(node) == 'Zone_t')
        vtx_size = PT.Zone.VertexSize(zone)
        idx_dim = len(vtx_size)
        pl = PT.get_np_value(last)

        wrong_id = np.zeros(pl.shape[1], bool)
        for i in range(idx_dim):
            wrong_id |= (pl[i] < 1) | (vtx_size[i] < pl[i])

        n_wrong_loc = wrong_id.sum()
        if (n_wrong := comm.allreduce(n_wrong_loc)) > 0:
            zrange = '[' + ', '.join(f"[1, {vtx_size[i]}]" for i in range(idx_dim)) + ']'
            return f"Some ids of this Vertex located subset are out of vertex range {zrange} ({n_wrong} found)"

    return OK

def invalid_elt_subset_id(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E309 - Invalid element subset values

    The values of element-located PointList (EdgeCenter, FaceCenter, CellCenter)
    should represent element ids of corresponding dimension.

    Erroneous tree example:

    zone Zone_t I4 [[125 320 0]]
    ├───ZoneType ZoneType_t "Unstructured"
    ├───TETRA_4 Elements_t I4 [10 0]
    │   └───ElementRange IndexRange_t I4 [1 320]
    ├───TRI_3 Elements_t I4 [5 0]
    │   └───ElementRange IndexRange_t I4 [321 512]
    ├───ZoneBC ZoneBC_t
    │   └───BC BC_t "BCWall"
    │       ├───GridLocation GridLocation_t "FaceCenter"
    │       └───PointList IndexArray_t I4 [[\033[32m460\033[m \033[91m560\033[m]] \033[91m# Maximal element id is 512\033[m
    └───ZoneSubRegion ZoneSubRegion_t
        ├───GridLocation GridLocation_t "CellCenter"
        ├───PointList IndexArray_t I4 [[\033[32m200 300 \033[91m400\033[m]] \033[91m# Location is CellCenter, but elt 400 is a face (TRI_3)\033[m
        └───Density DataArray_t R8 [1.013 1.014 1.013]
    """

    last = nodes[-1]
    if PT.get_name(last) == 'PointList' and (loc:=PT.Subset.GridLocation(nodes[-2])) != 'Vertex':
        # zone/base node can be at diffent level
        base = next(node for node in nodes if PT.get_label(node) == 'CGNSBase_t')
        zone = next(node for node in nodes if PT.get_label(node) == 'Zone_t')
        if PT.Zone.Type(zone) != 'Unstructured':
            return OK
         
        if loc == 'CellCenter':
            dim = PT.Base.CellDimension(base)
        else:
            dim = {'EdgeCenter' : 1, 'FaceCenter' : 2}[loc]

        pl = PT.get_np_value(last)[0]
        tgt_dim = -1*np.ones(pl.size, np.int8)
        for elt in utils.get_zone_elements(zone):
            elt_range = PT.Element.Range(elt)
            tgt_dim[(elt_range[0] <= pl) & (pl <= elt_range[1])] = PT.Element.Dimension(elt)

        ref, count = np.unique(tgt_dim, return_counts=True)
        dim_to_count = defaultdict(int, {d:c for d,c in zip(ref, count) if d!=dim})
        dim_to_count_g = comm.allreduce(dim_to_count, op = utils.defaultdict_sum)

        if -1 in dim_to_count_g:
            return f"No corresponding ElementRange for {dim_to_count_g[-1]} ids of this element located subset"
        elif len(dim_to_count_g) > 0:
            details = ''
            while (dim_to_count_g):
                key, val = dim_to_count_g.popitem()
                details += f"{key} ({val})"
                if len(dim_to_count_g) > 1:
                    details += ', '
                elif len(dim_to_count_g) == 1:
                    details += ' and '
            return f"Subset location is {loc}, but some ids reference entities of dimension {details}"


    return OK

def missing_opposite_join(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E310 - Missing opposite join

    For each GridConnectivity(1to1)_t node of type 'Abutting1to1' G, an opposite
    GridConnectivity(1to1)_t node G' should exist within the target zone of G, such that:
    - The GridLocation of G and G' are identical
    - The number of mesh entities of G and G' are identical
    - For each pair (local_id=i, opp_id=j) of G, the pair (local_id=j, opp_id=i) exists in G'

    Erroneous tree examples:

    Base CGNSBase_t I4 [2 2]
    ├───ZoneA Zone_t
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───\033[32mmatchAB\033[0m GridConnectivity_t \033[32m"ZoneB"\033[0m
    │           ├───GridConnectivityType GridConnectivityType_t \033[32m"Abutting1to1"\033[0m
    │           ├───GridLocation GridLocation_t "EdgeCenter"
    │           ├───PointList IndexArray_t I4 [[10 11 12]]
    │           └───PointListDonor IndexArray_t I4 [[7 8 9]]
    └───ZoneB Zone_t
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            ╵╴╴╴\033[91mMissing ZoneB to ZoneA join\033[0m

    Base CGNSBase_t I4 [2 2]
    ├───ZoneA Zone_t
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───\033[32mmatchAB\033[0m GridConnectivity_t \033[32m"ZoneB"\033[0m
    │           ├───GridConnectivityType GridConnectivityType_t \033[32m"Abutting1to1"\033[0m
    │           ├───GridLocation GridLocation_t "EdgeCenter"
    │           ├───PointList IndexArray_t I4 [[10 11 12]]
    │           └───PointListDonor IndexArray_t I4 [[7 8 9]]
    └───ZoneB Zone_t
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            └───matchBA GridConnectivity_t \033[32m"ZoneA"\033[0m
                ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
                ├───GridLocation GridLocation_t "EdgeCenter"
                ├───PointList IndexArray_t I4 \033[91m[[1 2 3]] # Not consistent with matchAB/PointListDonor\033[0m 
                └───PointListDonor IndexArray_t I4 [[12 11 10]]

    Correct example:

    Base CGNSBase_t I4 [2 2]
    ├───ZoneA Zone_t
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───\033[32mmatchAB\033[0m GridConnectivity_t \033[32m"ZoneB"\033[0m
    │           ├───GridConnectivityType GridConnectivityType_t \033[32m"Abutting1to1"\033[0m
    │           ├───GridLocation GridLocation_t "EdgeCenter"
    │           ├───PointList IndexArray_t I4 [[10 11 12]]
    │           └───PointListDonor IndexArray_t I4 [[7 8 9]]
    └───ZoneB Zone_t
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            └───matchBA GridConnectivity_t \033[32m"ZoneA"\033[0m
                ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
                ├───GridLocation GridLocation_t "EdgeCenter"
                ├───PointList IndexArray_t I4 [[9 8 7]]
                └───PointListDonor IndexArray_t I4 [[12 11 10]]
    """
    
    # Search matching JNs only once, on the full tree
    last = nodes[-1]
    if len(nodes) == 1 and not matching_jns_table.computed:
        matching_jns_table.compute(last, comm)

    # Report errors on GC_t nodes
    elif IS_GC_MATCH(last):
        path = '/'.join(n[0] for n in nodes[1:])
        opp_jns = matching_jns_table[path]
        if len(opp_jns) == 0:
            return "Opposite 1to1 GridConnectivity_t node not found in tree"
        elif len(opp_jns) > 1:
            return f"Several opposite 1to1 GridConnectivity_t node found (1 expected) : {opp_jns}"

    return OK

def non_symmetric_opposite_joins(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """W311 - Non symmetric opposite joins

    This rules extends E310 by adding this additional constraint of symmetry
    between two related Abutting1to1 GridConnectivity_t nodes G and G':

          PointList(G) == PointListDonor(G')     (or resp. PointRange)
      and PointListDonor(G) == PointList(G')     (or resp. PointRange)

    In other words, the pairs (local_id, opp_id) must be described in same order
    in the two related joins. This rule is not required by the CGNS standard,
    but some solvers or tools rely on it.

    Erroneous tree example:

    Base CGNSBase_t I4 [2 2]
    ├───ZoneA Zone_t
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───matchAB GridConnectivity_t "ZoneB"
    │           ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
    │           ├───GridLocation GridLocation_t "EdgeCenter"
    │           ├───PointList IndexArray_t I4 \033[32m[[10 11 12]]\033[0m
    │           └───PointListDonor IndexArray_t I4 \033[32m[[7 8 9]]\033[0m
    └───ZoneB Zone_t
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            └───matchBA GridConnectivity_t "ZoneA"
                ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
                ├───GridLocation GridLocation_t "EdgeCenter"
                ├───PointList IndexArray_t I4 \033[93m[[9 8 7]]\033[0m
                └───PointListDonor IndexArray_t I4 \033[93m[[12 11 10]] # Order is permuted\033[0m 

    Correct example:

    Base CGNSBase_t I4 [2 2]
    ├───ZoneA Zone_t
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───matchAB GridConnectivity_t "ZoneB"
    │           ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
    │           ├───GridLocation GridLocation_t "EdgeCenter"
    │           ├───PointList IndexArray_t I4 \033[32m[[10 11 12]]\033[0m
    │           └───PointListDonor IndexArray_t I4 \033[32m[[7 8 9]]\033[0m
    └───ZoneB Zone_t
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            └───matchBA GridConnectivity_t "ZoneA"
                ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
                ├───GridLocation GridLocation_t "EdgeCenter"
                ├───PointList IndexArray_t I4 \033[32m[[7 8 9]]\033[0m
                └───PointListDonor IndexArray_t I4 \033[32m[[10 11 12]]\033[0m
    """
    # Search matching JNs only once, on the full tree
    last = nodes[-1]
    if len(nodes) == 1 and not matching_jns_table.computed:
        matching_jns_table.compute(last, comm)

    elif IS_GC_MATCH(last):
        from maia.algo.dist.matching_jns_tools import _jn_is_symmetric_loc
        cur_path = '/'.join(n[0] for n in nodes[1:])
        opp_paths = matching_jns_table[cur_path]
        assert len(opp_paths) == 1 # Can not proceed if donor is not found
        opp_path = opp_paths[0]

        is_symm_loc = _jn_is_symmetric_loc(last, PT.find_node_from_path(nodes[0], opp_path))
        if not comm.allreduce(is_symm_loc, MPI.LAND):
            return f"Subsets ordering of matching GC_t node /{opp_path} differ"

    return OK


def duplicated_elts_connectivity(nodes:List[CGNSTree], comm:MPIComm) -> str:
    """E312 - Duplicated mesh entities

    Within a same zone, two different mesh entities should not be defined by
    the same list of vertices (or faces for NFACE_n elements).

    Erroneous tree examples:

    Zone Zone_t
    └───NG Elements_t I4 [22 0]
        ├───ElementRange IndexRange_t I4 [1 3]
        ├───ElementStartOffset DataArray_t I4 [0 3 4 7]
        └───ElementConnectivity DataArrray_t I4 [\033[32m1 3 4\033[0m  1 4 5 6  \033[91m3 4 1\033[0m] \033[91m# Already defined\033[0m

    Zone Zone_t
    ├───TRI_1 Elements_t I4 [5 0]
    │   ├───ElementRange IndexRange_t I4 [1 3]
    │   └───ElementConnectivity DataArrray_t I4 [1 2 3 \033[32m1 3 4\033[0m 1 4 5]
    └───TRI_2 Elements_t I4 [5 0]
        ├───ElementRange IndexRange_t I4 [4 5]
        └───ElementConnectivity DataArrray_t I4 [1 5 6 \033[91m3 4 1\033[0m] \033[91m # Already defined in TRI_1 section\033[0m
    """

    last = nodes[-1]
    if PT.get_label(last) != 'Zone_t':
        return OK

    all_elts = PT.Zone.get_ordered_elements(last)
    batches = [
        [e for e in all_elts if PT.Element.Type(e) not in ['NFACE_n']],
        [e for e in all_elts if PT.Element.Type(e) == 'NFACE_n'],
    ]

    tot_duplicated = 0
    all_duplicated = []
    for elts in batches:
        if len(elts) == 0:
            continue
        cnts = [MT.Element.connectivity(e) for e in elts]
        cnts = [abs(cnt) if PT.Element.Type(e) == 'NFACE_n' else cnt
                for e,cnt in zip(elts, cnts)]
        cnt = vs.concatenate(cnts, vs.OUTER_AXIS) if len(cnts) > 1 else cnts[0]

        # Compute hash using PROD (few collisions)
        # We do it on unsigned integers to have defined behaviour
        # in case of overflow
        ini_dtype = cnt.dtype
        u_dtype = {'i' : np.uint32, 'l' : np.uint64}[ini_dtype.char]
        cnt.values.dtype = u_dtype
        _hash = cnt.reduce(vs.ReduceOp.PROD)
        cnt.values.dtype = ini_dtype
        if ini_dtype.char == 'l':
            # uint64 can give very large dispersion -> reduce to uint32
            hash = np.empty(_hash.size, np.uint32)
            np.mod(_hash, 2**32, out=hash)
        else:
            hash = _hash
        # Uniform distribution should be suffisant if hash function is
        # reparted. In addition PDM does not manage hash values < 1 
        #distri = PDM.compute_weighted_distribution([hash], [np.ones(len(hash))], comm)
        distri = utils.compute_distri(hash, comm)
        # output distribution [0, last[ (size n_rank+1). To do binary search we
        # don't need the external bounds

        dest = np.searchsorted(distri[1:-1], hash, side='right')
        sort_idx = np.argsort(dest)

        # Phase 1 : Send hash to relevant rank for counting

        send_n = np.zeros(comm.size, np.int32)
        recv_n = np.empty(comm.size, np.int32)
        np.add.at(send_n, dest, 1)

        comm.Alltoall(send_n, recv_n) # Hash numbers

        recv_hash = np.empty(recv_n.sum(), hash.dtype)
        comm.Alltoallv((hash[sort_idx], send_n), (recv_hash, recv_n)) # Hash lists
        #ideal = comm.allreduce(recv_hash.size) / comm.size
        #diff = abs(recv_hash.size-ideal)/ideal
        #print(f"{comm.rank} N RECV HASH {recv_hash.size} ({'+' if recv_hash.size >= ideal else '-'}{100*diff:.3f}%)")

        # Detect hash appearing twice
        _, inv, counts = np.unique(recv_hash, return_inverse=True, return_counts=True)
        is_dupl = counts[inv] != 1

        if not comm.allreduce(is_dupl.any(), MPI.LOR):
            continue # Early exit if all hashes are unique

        initial_is_dupl_mpi = np.empty(hash.size, bool) # View on initial ranks, but sorted in MPI order
        comm.Alltoallv((is_dupl, recv_n), (initial_is_dupl_mpi, send_n))

        initial_is_dupl = np.empty_like(initial_is_dupl_mpi)
        initial_is_dupl[sort_idx] = initial_is_dupl_mpi   # View on initial ranks, unsorted (initial order)

        # Phase 2 : Filter values for collising hashes and send it for collision resolution
        # We already have sort order (to prepare MPI buffs), we just need to exclude
        # unique values
        selector = sort_idx[initial_is_dupl_mpi]
        filtered_cnt = vs.take(cnt, selector)

        elt_ranges = [PT.Element.Range(e) for e in elts]
        elt_distri = [MT.distribution_value(e, 'Element') for e in elts]
        starts = np.array([d[0]+r[0] for d,r in zip(elt_distri, elt_ranges)])
        stops  = np.array([d[1]+r[0] for d,r in zip(elt_distri, elt_ranges)])
        elt_ids = np_utils.multi_arange(starts, stops)[selector]

        # Reuse send_counts / recv_counts arrays
        send_n.fill(0)
        np.add.at(send_n, dest[initial_is_dupl], 1)

        comm.Alltoall(send_n, recv_n)

        send_counts = filtered_cnt.counts
        recv_counts = np.zeros(recv_n.sum(), send_counts.dtype)
        recv_ids    = np.zeros(recv_n.sum(), elt_ids.dtype)
        comm.Alltoallv((send_counts, send_n), (recv_counts, recv_n))
        comm.Alltoallv((elt_ids, send_n), (recv_ids, recv_n))

        # Reuse again send_n / recv_n (update inplace)
        for n_item, counts in zip((send_n, recv_n), (send_counts, recv_counts)):
            start = 0
            for i in range(comm.size):
                end = start + n_item[i]
                n_item[i] = counts[start:end].sum()
                start = end

        recv_vals = np.empty(recv_n.sum(), filtered_cnt.dtype)
        comm.Alltoallv((filtered_cnt.values, send_n), (recv_vals, recv_n))
        recv_vstride = vs.from_counts(recv_counts, recv_vals)

        # Here we assume that collisions are rare enought to do this
        # with a pure python loop
        recv_vstride._inner_sort()
        counter = defaultdict(list)
        for id,blk in zip(recv_ids, recv_vstride):
            counter[blk.tobytes()].append(id)

        duplicated_ids = [set(val) for val in counter.values() if len(val) > 1]
        
        # Gather duplicates on rank 0
        if (n_duplicated := comm.allreduce(len(duplicated_ids))) > 0:
            tot_duplicated += n_duplicated
            all_duplicated_batch = comm.reduce(duplicated_ids[:3], root=0)
            if comm.rank == 0:
                all_duplicated += all_duplicated_batch
                
    
    # Loop completed, return duplicated
    if len(all_duplicated) > 0: # Only rank 0
        # Here we retrieve the name of elements to display more information
        allstops = [PT.Element.Range(e)[1] for e in all_elts]
        allnames = [PT.get_name(e) for e in all_elts]
        sub_msgs = list()
        for ids in all_duplicated[:3]:
            sub = "{"
            for id in ids:
                j = 0
                while id > allstops[j]:
                    j += 1
                sub += f"{id} ({allnames[j]}), "
            sub_msgs.append(sub[:-2] + '}')

        msg = f"Some mesh elements have the same definition : {', '.join(sub_msgs)}"
        if tot_duplicated > 3:
            msg += f' ... ({tot_duplicated} groups detected)'
        return msg

    return OK



_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)

DNODE_RULES = {func[1].__doc__[:4] : func[1] for func in _funcs}
assert len(DNODE_RULES) == len(_funcs)
