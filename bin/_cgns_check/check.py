import h5py
import math
import os
from pathlib import Path
from mpi4py  import MPI

from maia.typing import *

import maia.pytree as PT
import maia.pytree.core.graph as PTG

OK = ''

class Colors:
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    LINK = '\033[36m'
    UNDERLINE = '\033[4m'

class FakeArray:
    """ This class mimics a ndarray by providing shape, dtype and size,
    but no data is actually stored """
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = math.prod(shape)

def lazy_load_cgns(filename:Path, exclude:List[str]) -> CGNSTree:
    # Here we rewritted a CGNS reader to prevent crashes in the following cases:
    # - if b_kind is not MT but ' data' does not exists (use MT value)
    # - if a link can not be followed (remove node)
    
    def tree_creator(hdf_node, py_children):
        # The constructor callback used by depth_first_build
        # hdf_node is the current node from graph 1 (=hdf graph)
        # py_children is the list of already build children for graph 2 (pytree)

        attrs = hdf_node.attrs
        name = hdf_node.name.split('/')[-1]
        label = attrs['label'].decode() if 'label' in attrs else ''
        kind = attrs['type']            if 'type'  in attrs else 'MT'
        value = None

        if kind == b'LK':
            label = 'Linked_t'

        # Lazy load data
        if kind not in {b'MT', b'LK'} and ' data' in hdf_node:
            _data = hdf_node[' data']
            if label not in {'DataArray_t', 'IndexArray_t'}:
                value = _data[()].T
                if kind == b'C1':
                    value.dtype = 'S1' 
            else:
                shape = _data.shape[::-1]
                dtype = _data.dtype
                value = FakeArray(shape, dtype)
                # Special case of ElementStartOffset: we need to load the last
                # value, to compare it later with ElementConnectivity.size
                if name == 'ElementStartOffset':
                    value.last_value = _data[-1] #type:ignore

        # Remove intermediary link level when in children when
        # creating current node
        nolink_children = []
        for child in py_children:
            if child[3] == 'Linked_t':
                n_sub_child = len(child[2])
                if n_sub_child == 0:
                    pass # Link is broken --> do nothing
                elif n_sub_child == 1:
                    sub_node = child[2][0]
                    sub_node[0] = child[0] # Use name from src node and not from target
                    nolink_children.append(sub_node)
                else:
                    raise RuntimeError # Link should note have more than 1 child
            else:
                nolink_children.append(child)

        return [name, value, nolink_children, label]

    with h5py.File(filename) as f:
        t = PTG.depth_first_build(HDF5GraphAdaptor(f['/'], exclude), tree_creator)
        # Update root node name / label
        t[0] = 'CGNSTree'
        t[3] = 'CGNSTree_t'

    return t

def _keep_subset_node(node, additional_child_to_keep=[], donor=False):
    """ Cleanup node (remove unecessary child), add #Size info and return
    True if node must be keep """
    suffix = 'Donor' if donor else ''
    names_to_keep = ['GridLocation', 'PointList', 'PointRange'] + additional_child_to_keep
    PT.keep_children_from_predicate(node, PT.pred.name_in(names_to_keep))
    pl_node = PT.get_child_from_name(node, f'PointList{suffix}')
    pr_node = PT.get_child_from_name(node, f'PointRange{suffix}')
    one_patch = (pl_node is not None) ^ (pr_node is not None)
    if not one_patch:
        return False # Skip nodes having both PointRange and PointList or None
    if pl_node is not None:
        if not isinstance(val:=pl_node[1], FakeArray):
            return False # Skip nodes having MT PointList value
        PT.new_node(f'PointList{suffix}#Size', 'DataArray_t', val.shape, parent=node) # Required for read
    return True

def fill_cgns(tree: CGNSTree, filename:Path, exclude:List[str], comm:MPIComm):
    from maia.io import cgns_io_tree as IOT
    
    # Again we rewritted fill_cgns method to have something more error-tolerant
    # and to read only necessary data (ie. not fields)
    # The idea is to select only the nodes to be keep and then call fill_size_tree
    # with these nodes
    
    # Cleaning tree 
    for base, zone in PT.get_children_from_predicates(tree, 'CGNSBase_t/Zone_t', ancestors=True):
        zone_path = f'{PT.get_name(base)}/{PT.get_name(zone)}'
        names_to_keep = ['ZoneType']

        # GridCoordinates_t: keep node if shape is consistant with zone size
        for grid_co in PT.iter_children_from_label(zone, 'GridCoordinates_t'):
            PT.rm_children_from_name(grid_co, 'CoordinateTransform')
            for coord in PT.get_children_from_label(grid_co, 'DataArray_t'):
                if not (isinstance(val:=coord[1], FakeArray) and val.shape == PT.Zone.VertexSize(zone)):
                    break
            else: # Loop did not break => all DataArray OK => load
                names_to_keep.append(PT.get_name(grid_co))

        # Elements_t : keep at least ElementRange if well defined, and keep 
        # connectivity arrays if possible
        for elt in PT.iter_children_from_label(zone, 'Elements_t'):

            # Completly skip element if ElementRange is not valid
            to_keep = ['ElementRange']
            try:
                if (elt_size := PT.Element.Size(elt)) < 0:
                    continue
            except Exception:
                continue

            if PT.Element.Type(elt) in ['NGON_n', 'NFACE_n', 'MIXED']:
                # Nodes with ESO are complicated -> load by hand 
                ec_size = None
                if (eso:=PT.get_child_from_name(elt, 'ElementStartOffset')) is not None:
                    if isinstance(val := eso[1], FakeArray) and val.shape == (elt_size+1,):
                        ec_size = val.last_value # type:ignore
                        to_keep.append('ElementStartOffset')

                    # ESO will be loaded -> erase FakeArray value, otherwise ElementConnectivity
                    # distribution computing is trigered to early
                    PT.set_value(eso, None)

                if (ec:=PT.get_child_from_name(elt, 'ElementConnectivity')) is not None:
                    if isinstance(val := ec[1], FakeArray) and val.shape == (ec_size,):
                        # We need to analyse ESO to tell if EC is loadable, so load ESO
                        # ESO exists otherwise we would have (val.shape) != (None,)
                        from maia.utils import par_utils
                        distri = par_utils.uniform_distribution(elt_size, comm)
                        dn = distri[1] - distri[0]
                        DS = [[0],[1],[dn+1],[1], [distri[0]],[1],[dn+1],[1], [elt_size+1], [0]]
                        #     ^MMRY               ^FILE                       ^GLOB         ^FLAG

                        eso_path = f'{zone_path}/{PT.get_name(elt)}/ElementStartOffset'
                        eso_node = PT.find_node_from_path(tree, eso_path)
                        IOT.load_partial(str(filename), tree, {eso_path:DS}, comm)
                        eso_val = PT.get_np_value(eso_node)

                        is_valid_eso = comm.bcast(eso_val[0]  == 0,       root=0)           and \
                                       comm.bcast(eso_val[-1] == ec_size, root=comm.size-1) and \
                                       comm.allreduce((eso_val[:-1] <= eso_val[1:]).all(), MPI.LAND)

                        if is_valid_eso:
                            PT.new_node('ElementConnectivity#Size', 'DataArray_t', [ec_size], parent=elt)
                            to_keep.extend(['ElementConnectivity', 'ElementConnectivity#Size'])

                        PT.set_value(eso_node, None) # Reput None in ESO (see above)

            else:
                expt_ec_shape = (PT.Element.NVtx(elt)*elt_size,)
                if (ec:=PT.get_child_from_name(elt, 'ElementConnectivity')) is not None:
                    if isinstance(val := ec[1], FakeArray) and val.shape == expt_ec_shape:
                        to_keep.append('ElementConnectivity')

            if (pe:=PT.get_child_from_name(elt, 'ParentElements')) is not None:
                if isinstance(val := pe[1], FakeArray) and val.shape == (elt_size, 2):
                    to_keep.append('ParentElements')
                
            PT.keep_children_from_predicate(elt, PT.pred.name_in(to_keep))
            names_to_keep.append(PT.get_name(elt))

        
        # Containers: don't load related ZSR or full containers (nothing more to check),
        # disable containers having PR *and* PL, and don't load fields
        is_container = PT.pred.label_in(['ZoneSubRegion_t', 'FlowSolution_t', 'DiscreteData_t'])
        for ctn in PT.iter_children_from_predicate(zone, is_container):
            if _keep_subset_node(ctn):
                names_to_keep.append(PT.get_name(ctn))

        # BCs and BCDataSets : same as containers
        for zbc in PT.iter_children_from_label(zone, 'ZoneBC_t'):
            bc_names_to_keep = list()
            for bc in PT.iter_children_from_label(zbc, 'BC_t'):
                bcds_names_to_keep = list()
                for bcds in PT.iter_children_from_label(bc, 'BCDataSet_t'):
                    if _keep_subset_node(bcds):
                        bcds_names_to_keep.append(PT.get_name(bcds))
                if _keep_subset_node(bc, bcds_names_to_keep):
                    bc_names_to_keep.append(PT.get_name(bc))

            PT.keep_children_from_predicate(zbc, PT.pred.name_in(bc_names_to_keep))
            names_to_keep.append(PT.get_name(zbc))

        # GridConnectivity(1to1)_t: same, with additional treatment for Donor nodes
        for zgc in PT.iter_children_from_label(zone, 'ZoneGridConnectivity_t'):
            gc_names_to_keep = list()
            for gc in PT.iter_children_from_predicate(zgc, PT.pred.IS_GC):
                is1to1 = PT.GridConnectivity.is1to1(gc)
                to_keep = ['GridConnectivityProperty', 'GridConnectivityType']
                # Carefull ! here we want a DA that is not distributed (TODO)
                if is1to1:
                    to_keep += ['PointListDonor', 'PointRangeDonor']
                if not _keep_subset_node(gc, to_keep):
                    continue
                if is1to1:
                    to_keep.append('PointList#Size') # Just created
                    if not _keep_subset_node(gc, to_keep, True):
                        # If donor nodes are misformed, juste remove them
                        PT.rm_children_from_name(gc, 'PointListDonor')
                        PT.rm_children_from_name(gc, 'PointRangeDonor')

                gc_names_to_keep.append(PT.get_name(gc))

            PT.keep_children_from_predicate(zgc, PT.pred.name_in(gc_names_to_keep))
            names_to_keep.append(PT.get_name(zgc))



        
        PT.keep_children_from_predicate(zone, PT.pred.name_in(names_to_keep))

    for base in PT.get_children_from_predicates(tree, 'CGNSBase_t'):
        PT.keep_children_from_label(base, 'Zone_t')

    # Effective loading for remaning arrays
    IOT.fill_size_tree(tree, filename, comm)



class CGNSChecker:
    """ A visitor that apply check rules to input tree.
    If comm is None, rules must have the signature rule(nodes).
    Otherwise, they must have the signature rule(nodes, comm). """
    def __init__(self, rules, comm=None) -> None:
        self.rules = rules
        self.comm = comm
    
    def pre(self, nodes: List[CGNSTree]):
        raised = []
        args = [nodes, self.comm] if self.comm is not None else [nodes]

        for rule_id, rule_fn in self.rules.items():
            try:
                out = rule_fn(*args)
            except Exception:
                raised.append(rule_id)
                out = OK
            if out != OK:
                path = '/' if len(nodes) == 1 else '/'.join(n[0] for n in nodes)[8:]
                color = Colors.FAIL if rule_id.startswith('E') else Colors.WARNING
                if self.comm is None or self.comm.rank == 0:
                    print(f"{path}: {color}{rule_id}{Colors.ENDC} {out}")
        if len(raised) > 0:
            path = '/' if len(nodes) == 1 else '/'.join(n[0] for n in nodes)[8:]
            if self.comm is None or self.comm.rank == 0:
                print(f"{path}: {Colors.HEADER}Unable to check {','.join(raised)} due to other errors{Colors.ENDC}")


class HDF5GraphAdaptor:
    """ A class exposing the 'graph interface' for hdf files in order
    to use graph iterators """
    def __init__(self, root, exclude_l=[]):
        self.root = root
        self.exclude_l = exclude_l
    def root_iterator(self) -> PTG.list_iterator_type:
        return iter([self.root])
    def child_iterator(self, node) -> PTG.list_iterator_type:
        # Links are automatically traversed by the iterator if they exist
        # Otherwise, no exception is raised because .values() returns None for
        # 'broken' links, and other nodes are still visited.
        # This is exactly what we want; we will report 'broken' links during checks
        return (g for g in node.values() if (isinstance(g, h5py.Group) and g.name not in self.exclude_l))

class HDFChecker:
    """ Graph visitor used to collect link information """
    def __init__(self, rules):
        self.rules = rules
        self.names = [''] # Path of node in main structure

    def pre(self, node):
        if node.name == '/': # Store current file to check external link
            self.file = node.file
            self.filedir = Path(self.file.filename).resolve().parent
        raised = []
        for rule_id, rule_fn in self.rules.items():
            try:
                out = rule_fn(node)
            except Exception:
                raised.append(rule_id)
                out = OK
            if out != OK:
                color = Colors.FAIL if rule_id.startswith('E') else Colors.WARNING
                path = '/'.join(self.names) if len(self.names) > 1 else '/'
                if node.file != self.file: # Internal link not yet managed
                    # Here we add the linked file / path in output
                    linked_file = Path(node.file.filename)
                    rel_file = os.path.relpath(linked_file.resolve(), self.filedir)
                    path += f' {Colors.LINK}(-> {rel_file}::{node.name}){Colors.ENDC}'
                print(f"{path}: {color}{rule_id}{Colors.ENDC} {out}")
        if len(raised) > 0:
            print(f"{node.name}: {Colors.HEADER}Exception raised when checking {Colors.ENDC}{','.join(raised)}")
    
    def down(self, parent, child):
        is_link = parent.file != child.file # Internal link not yet managed (maybe child.parent != parent)
        if not is_link:
            self.names.append(child.name.split('/')[-1])
    def up(self, child, parent):
        is_link = parent.file != child.file # Internal link not yet managed (maybe child.parent != parent)
        if not is_link:
            self.names.pop()

def run_stage_1(filename:Path, ignore_list:List[str], exclude_list:List[str]) -> bool:

    from .rules1 import FILE_RULES, GROUP_RULES
    # First pass to test file rules (errors are fatal)
    for rule_id, rule_fn in FILE_RULES.items():
        if (out := rule_fn(filename)) != OK:
            msg = f"Execution aborted due to {Colors.FAIL}fatal error {rule_id}{Colors.ENDC}:" \
                  f" {out}"
            print(msg)
            return False
    
    # Now test hdf rules using DFS traversal
    rules = {key:val for key, val in GROUP_RULES.items() if key not in ignore_list}
    with h5py.File(filename) as f:
        PTG.depth_first_search(HDF5GraphAdaptor(f['/'], exclude_list), HDFChecker(rules))

    return True


def run_stage_2(tree:CGNSTree, ignore_list:List[str]) -> bool: 
    from .rules2 import NODE_RULES
    # Prepare tree visitor
    rules = {key:val for key, val in NODE_RULES.items() if key not in ignore_list}
    PT.visit(tree, CGNSChecker(rules), ancestors=True)

    return True

def run_stage_3(tree:CGNSTree, ignore_list:List[str], comm:MPIComm) -> bool: 
    from .rules3 import DNODE_RULES
    rules = {key:val for key, val in DNODE_RULES.items() if key not in ignore_list}
    PT.visit(tree, CGNSChecker(rules, comm), ancestors=True)
    return True

def check(args):

    # Format exclude list to start as hdf path
    if '/' in args.exclude:
        return
    for i,path in enumerate(args.exclude):
        if path[-1] == '/': # Remove last '/' if provided
            path = path[:-1]
        if path.startswith('CGNSTree'): # Remove CGNSTree if used
            args.exclude[i] = path[8:]
        elif path[0] != '/': # Add first '/' if missing
            args.exclude[i] = '/' + path

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        st = run_stage_1(args.filename, args.ignore, args.exclude)
    else:
        st = False
    st = comm.bcast(st, root=0)
    if not st:
        exit(1)

    # For now stage 2 is serial also // We should distribute
    # checks zones over ranks
    if comm.rank == 0:
        # Partial load of cgnsfile : heavy data are not loaded
        tree = lazy_load_cgns(args.filename, args.exclude)
        st = run_stage_2(tree, args.ignore)
    else:
        tree = None

    if args.lazy:
        exit()

    tree = comm.bcast(tree, root=0)
    fill_cgns(tree, args.filename, args.exclude, comm)
    st = run_stage_3(tree, args.ignore, comm)