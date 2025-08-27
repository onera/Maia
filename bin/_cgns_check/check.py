import h5py
import math
import os
from pathlib import Path

from maia.typing import *

import maia.pytree.graph as PTG

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

def lazy_load_cgns(filename:Path) -> CGNSTree:
    # Here we rewritted a CGNS reader to prevent crashes in the following cases:
    # - if b_kind is not MT but ' data' does not exists (use MT value)
    # - if a link can not be followed (remove node)
    
    def tree_creator(hdf_node, py_children):
        # The constructor callback used by depth_first_build
        # hdf_node is the current node from graph 1 (=hdf graph)
        # py_children is the list of already build children for graph 2 (pytree)

        name = hdf_node.name.split('/')[-1]
        label = hdf_node.attrs['label'].decode()
        kind = hdf_node.attrs['type']
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
        from maia.pytree.graph.build import depth_first_build
        t = depth_first_build(HDF5GraphAdaptor(f['/']), tree_creator)
        # Update root node name / label
        t[0] = 'CGNSTree'
        t[3] = 'CGNSTree_t'

    return t


def lazy_load_cgns3(filename:Path) -> CGNSTree:
    class TreeCreator:
        """ Reimplement the HDF read to be more error-tolerant """
        def __init__(self):
            self.tree = ['CGNSTree', None, [], 'CGNSTree_t']
            self.stack = [self.tree]
            self.names = []
            self.is_lk = False

        def pre(self, node):
            print("Pre for ", node, '(', self.is_lk, ')' ' --->', '/'.join(self.names))

            if node.name == '/':
                return

            parent = self.stack[-1]
            name = node.name.split('/')[-1]
            label = node.attrs['label'].decode()
            kind = node.attrs['type']
            value = None

            # Mark linked nodes so we can detect 
            if kind == b'LK':
                #label = 'Linked_t'
                if node.get(' link') is None: # Broken link
                    print("BROKEN", node)
                    self.stack.append(None)
                    return PTG.algo.step.over
            
            # Data
            if kind not in {b'MT', b'LK'} and ' data' in node:
                _data = node[' data']
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
                        value.last_value = _data[-1]

            # TODO :: manage broken links
            if self.is_lk:
                # Update parent node
                print("LINK", node)
                parent[1] = value
                parent[3] = label
                pass
            else:
                # Create new node
                py_node = [name, value, [], label]
                parent[2].append(py_node)
                self.stack.append(py_node)


            # Si pas is_lk False --> classique
            # Sinon : ne pas ajouter le noeud mais updater Label / type 
            # Si noeud existe pas ??
            #print('/'.join(self.names), node.file.filename.split('/')[-1])

        #def post(self, node):
            #print("post for", node, '(', self.is_lk, ')')
            #if not self.is_lk:
                #self.stack.pop()
            
        def down(self, parent, child):
            #print("down from", parent, "to", child)
            is_link = parent.file != child.file # Internal link not yet managed (maybe child.parent != parent)
            self.is_lk = is_link
            if not is_link:
                self.names.append(child.name.split('/')[-1])

        def up(self, child, parent):
            is_link = parent.file != child.file # Internal link not yet managed (maybe child.parent != parent)
            self.is_lk = is_link
            #print("up from", child, "to", parent, '(', is_link, ')')
            if not is_link:
                self.names.pop()
                self.stack.pop()

    tc = TreeCreator()
    print("\n\n\n")
    with h5py.File(filename) as f:
        #f['/'].visit(printname)
        pass
        PTG.algo.depth_first_search(HDF5GraphAdaptor(f['/']), tc)
    import maia.pytree as PT
    PT.print_tree(tc.tree)
    quit()
    return tc.tree

def lazy_load_cgns2(filename:Path) -> CGNSTree:
    class TreeCreator:
        """ Reimplement the HDF read to be more error-tolerant """
        def __init__(self):
            self.tree = ['CGNSTree', None, [], 'CGNSTree_t']
            self.stack = [self.tree]
        def pre(self, node):
            if node.name == '/':
                return

            parent = self.stack[-1]
            name = node.name.split('/')[-1]
            label = node.attrs['label'].decode()
            kind = node.attrs['type']
            value = None

            print(name)
            if name == ' link':
                print("Exti from ", node)
                return
            if kind == b'LK':
                print("PARENT IS", parent)
                print("NOW NODE", node, "from file", node.file)
                node = node[' link']
                label = node.attrs['label'].decode()
                kind = node.attrs['type']
                print("NOW NODE", node, "from file", node.file)
                #node = node[]
            # Data
            if kind != b'MT' and ' data' in node:
                _data = node[' data']
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
                        value.last_value = _data[-1]

            py_node = [name, value, [], label]
            parent[2].append(py_node)

            self.stack.append(py_node)

        def post(self, node):
            print("OUT from", node, '(file', node.file, ')')
            self.stack.pop()

    tc = TreeCreator()
    with h5py.File(filename) as f:
        #f['/'].visit(printname)
        PTG.algo.depth_first_search(HDF5GraphAdaptor(f['/']), tc)
    return tc.tree

def lazy_load_cgnsO(filename:Path) -> CGNSTree:

    def load_data(names, labels, hdf_dataset):
        """ We must load the node value when print_tree will try to display it, ie:
        - if value are characters
        - if value size is <= 9
        """
        need_true_value = {'CGNSBase_t', 'Zone_t', 'Elements_t', 'IndexRange_t', '"int[IndexDimension]"',}
        # Types non MT / C1
        if hdf_dataset is None:
            return True
        else:
            # Todo : improve rule here
            ds_size = math.prod(hdf_dataset.shape)
            #return hdf_dataset.dtype.char == 'b' or ds_size <= 9
            #return hdf_dataset.dtype.char == 'b' or labels[-1] not in {'DataArray_t', 'IndexArray_t'}
            return labels[-1] not in {'DataArray_t', 'IndexArray_t'}

    def noload_fn(node, parent, hdf_dataset):
        """ If not is not loaded (cause it's heavy), create a FakeArray object to display
        it's shape and dtype """
        if hdf_dataset is not None:
            shape = hdf_dataset.shape[::-1]
            dtype = hdf_dataset.dtype
            node[1] = FakeArray(shape, dtype)

            # Special case of ElementStartOffset: we need to load the last
            # value, to compare it later with ElementConnectivity.size
            h5py_dset = h5py.Dataset(hdf_dataset)
            if h5py_dset.parent.name.endswith('ElementStartOffset'):
                node[1].last_value = h5py_dset[-1]
        parent[2].append(node)

    from maia.io.hdf._hdf_cgns import load_tree_partial
    return load_tree_partial(str(filename), load_data, noload_fn) #type:ignore

class CGNSChecker:
    """ A visitor for depth_first_search that collect the paths of UserDefinedData nodes """
    def __init__(self, rules) -> None:
        self.rules = rules

    """
    def pre(self, nodes: List[CGNSTree]):
        # Class impl
        for rule in self.rules:
            if not (rule.code in self.ignore_list or rule.check(nodes)):
                path = '/'.join(n[0] for n in nodes)
                color = Colors.FAIL if rule.code.startswith('E') else Colors.WARNING
                print(f"{path}: {color}{rule.code}{Colors.ENDC} {rule.out}")
    """
    
    def pre(self, nodes: List[CGNSTree]):
        # Pure fn impl
        raised = []

        for rule_id, rule_fn in self.rules.items():
            try:
                out = rule_fn(nodes)
            except Exception:
                raised.append(rule_id)
                out = ''
            if out != '':
                path = '/'.join(n[0] for n in nodes)
                color = Colors.FAIL if rule_id.startswith('E') else Colors.WARNING
                print(f"{path}: {color}{rule_id}{Colors.ENDC} {out}")
        if len(raised) > 0:
            path = '/'.join(n[0] for n in nodes)
            print(f"{path}: {Colors.HEADER}Exception raised when checking {Colors.ENDC}{','.join(raised)}")


class HDF5GraphAdaptor:
    """ A class exposing the 'graph interface' for hdf files in order
    to use graph iterators """
    def __init__(self, root):
        self.root = root
    def root_iterator(self) -> PTG.utils.list_iterator:
        return iter([self.root])
    def child_iterator(self, node) -> PTG.utils.list_iterator:
        # Links are automatically traversed by the iterator if they exist
        # Otherwise, no exception is raised because .values() returns None for
        # 'broken' links, and other nodes are still visited.
        # This is exactly what we want; we will report 'broken' links during checks
        return (g for g in node.values() if isinstance(g, h5py.Group))

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
                out = ''
            if out != '':
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

def run_stage_1(filename:Path, ignore_list:List[str]) -> bool:

    from .rules1 import FILE_RULES, GROUP_RULES
    # First pass to test file rules (errors are fatal)
    for rule_id, rule_fn in FILE_RULES.items():
        if (out := rule_fn(filename)) != '':
            msg = f"Execution aborted due to {Colors.FAIL}fatal error {rule_id}{Colors.ENDC}:" \
                  f" {out}"
            print(msg)
            return False
    
    # Now test hdf rules using DFS traversal
    rules = {key:val for key, val in GROUP_RULES.items() if key not in ignore_list}
    with h5py.File(filename) as f:
        PTG.algo.depth_first_search(HDF5GraphAdaptor(f['/']), HDFChecker(rules))

    return True


def run_stage_2(filename:Path, ignore_list:List[str]) -> bool: 
    from .rules2 import NODE_RULES
    # Partial load of cgnsfile : heavy data are not loaded
    tree = lazy_load_cgns(filename)
    # Prepare tree visitor
    rules = {key:val for key, val in NODE_RULES.items() if key not in ignore_list}
    PTG.cgns.depth_first_search(tree, CGNSChecker(rules), depth='all')

    return True

def check(args):
    ignore_list = args.ignore.split(',')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        st = run_stage_1(args.filename, ignore_list)
    else:
        st = False
    st = comm.bcast(st, root=0)
    if not st:
        exit(1)

    run_stage_2(args.filename, ignore_list)