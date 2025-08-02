import h5py
import math
from pathlib import Path

from maia.typing import *

import maia.pytree.graph as PTG

class Colors:
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def lazy_load_cgns(filename:Path) -> CGNSTree:
    class FakeArray:
        """ This class mimics a ndarray by providing shape, dtype and size,
        but no data is actually stored """
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.size = math.prod(shape)

    def load_data(names, labels, hdf_dataset):
        """ We must load the node value when print_tree will try to display it, ie:
        - if value are characters
        - if value size is <= 9
        """
        if hdf_dataset is None:
            return True
        else:
            # Todo : improve rule here
            ds_size = math.prod(hdf_dataset.shape)
            return hdf_dataset.dtype.char == 'b' or ds_size <= 9

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

    def pre(self, node):
        raised = []
        for rule_id, rule_fn in self.rules.items():
            try:
                out = rule_fn(node)
            except Exception:
                raised.append(rule_id)
                out = ''
            if out != '':
                color = Colors.FAIL if rule_id.startswith('E') else Colors.WARNING
                print(f"{node.name}: {color}{rule_id}{Colors.ENDC} {out}")
        if len(raised) > 0:
            print(f"{node.name}: {Colors.HEADER}Exception raised when checking {Colors.ENDC}{','.join(raised)}")


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
        PTG.algo.depth_first_search(HDF5GraphAdaptor(f), HDFChecker(rules))

    return True


def run_stage_2(filename:Path, ignore_list:List[str]) -> bool: 
    from .rules2 import NODE_RULES
    # Partial load of cgnsfile : heavy data are not loaded
    # TODO : we need to rewrite reader, otherwise some errors (as missing links) become fatal
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