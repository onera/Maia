import sys
import maia.pytree as PT
import numpy as np
import doctest
import importlib
import pytest

modules_to_test = [
    'maia.pytree.yaml.parse_yaml_cgns',
    'maia.pytree.yaml.parse_cgns_yaml',
    'maia.pytree.walk.remove_nodes',
    'maia.pytree.walk.walkers_api',
    'maia.pytree.sids.node_inspect',
    'maia.pytree.sids.adjust',
    'maia.pytree.node.print',
    'maia.pytree.node.presets',
    'maia.pytree.node.create',
    'maia.pytree.node.access',
    'maia.pytree.logical_op',
    'maia.pytree.compare',
    ]

def collect_doctests():
    tests = []
    for module_name in modules_to_test:
        module = importlib.import_module(module_name)
        finder = doctest.DocTestFinder()
        for dt in finder.find(module):
            # Injection des variables globales pour que les exemples s'exécutent correctement
            dt.globs.update({'sys': sys, 'PT': PT, 'np': np})
            tests.append((module_name, dt))
    return tests

@pytest.mark.parametrize("module_name,doc_test", collect_doctests())
def test_individual_doctest(module_name, doc_test):
    runner = doctest.DocTestRunner(verbose=False)
    runner.run(doc_test)
    failure_count, _ = runner.summarize(verbose=False)
    assert failure_count == 0, f"Doctest failed in module {module_name}, doc: {doc_test.name}"
