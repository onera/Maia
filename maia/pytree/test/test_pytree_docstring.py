import sys, re
import maia.pytree as PT
import numpy as np
import doctest
import importlib
import pytest

# List of modules to collect doctests from
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

# Monkey-patch PT.print_tree to inject default parameters: out=sys.stdout and colors=False.
_original_print_tree = PT.print_tree

def new_print_tree(*args, **kwargs):
    # Set default for 'out' if not provided
    kwargs.setdefault('out', sys.stdout)
    # Set default for 'colors' if not provided
    kwargs.setdefault('colors', False)
    return _original_print_tree(*args, **kwargs)

# Replace the original function with the new wrapper
PT.print_tree = new_print_tree

def apply_hooks(doc_test):
    """
    Apply hooks on each doctest.Example in the given DocTest.

    If the example source (after stripping leading whitespace) starts with "PT.new_", 
    modify it to insert "_ = ".
    """
    for example in doc_test.examples:
        code = example.source
        stripped_code = code.lstrip()
        if stripped_code.startswith("PT.new_"):
            # !!!Preserve the original indentation!!!
            indentation = code[:len(code) - len(stripped_code)]
            example.source = f"{indentation}_ = {stripped_code}"
    return doc_test

def collect_doctests():
    """
    Function to collect all doctests from the specified modules.

    For each module, we import it, use doctest.DocTestFinder to find all docstrings,
    update the globals with required variables, and apply our hooks.
    """
    tests = []
    for module_name in modules_to_test:
        module = importlib.import_module(module_name)
        finder = doctest.DocTestFinder()
        for dt in finder.find(module):
            dt.globs.update({'sys': sys, 'PT': PT, 'np': np})
            dt = apply_hooks(dt)
            tests.append((module_name, dt))
    return tests

class NoWhitespaceOutputChecker(doctest.OutputChecker):
    """
    A custom OutputChecker that ignores whitespace differences.
    """
    def check_output(self, want, got, optionflags):
        """
        Check if the output matches the expected output, ignoring whitespace differences.
        """
        normalized_want = ''.join(want.split())
        normalized_got  = ''.join(got.split())
        return normalized_want == normalized_got

WHITESPACE_PATTERNS = [
    # Patterns for which we want (yes example.want) to ignore whitespace differences
    r"DiffReport\(",
    r"CartesianCoordinates\(CoordinateX",
    r"PeriodicValues\(RotationCenter",
    r"\['GridLocation',\s*array\(\[",
]

class ConditionalWhitespaceOutputChecker(doctest.OutputChecker):
    """
    A custom OutputChecker that ignores whitespace differences for specific patterns.
    """
    def __init__(self, patterns=None):
        super().__init__()
        self.patterns = patterns or []
    
    def check_output(self, want, got, optionflags):
        """
        Check if the output matches the expected output, ignoring whitespace differences
        for specific patterns.
        """
        if any(re.search(pattern, want) for pattern in self.patterns):
            normalized_want = ''.join(want.split())
            normalized_got  = ''.join(got.split())
            return normalized_want == normalized_got
        else:
            return super().check_output(want, got, optionflags)

@pytest.mark.parametrize("module_name,doc_test", collect_doctests())
def test_individual_doctest(module_name, doc_test):
    """
    Execute an individual doctest and treat it as a separate test.
    """
    # 1. Default runner
    # runner = doctest.DocTestRunner(verbose=False)
    # 2. Universal NoWhitespaceOutputChecker runner
    # runner = doctest.DocTestRunner(
    #     verbose=False,
    #     checker=NoWhitespaceOutputChecker()
    # )
    # 3. Conditional WhitespaceOutputChecker runner
    runner = doctest.DocTestRunner(
        verbose=False,
        checker=ConditionalWhitespaceOutputChecker(patterns=WHITESPACE_PATTERNS)
    )
    runner.run(doc_test)
    failure_count, _ = runner.summarize(verbose=False)
    assert failure_count == 0, f"Doctest failed in module {module_name}, doc: {doc_test.name}"