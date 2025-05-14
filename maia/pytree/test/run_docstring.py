import sys
import re
import os
import importlib
import doctest
import maia.pytree as PT
import numpy as np

# --- Configuration ---
script_dir = os.path.abspath(os.path.dirname(__file__))
MAIA_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
MAIA_PYTREE_PATH = os.path.join(MAIA_BASE_DIR, "maia", "pytree")
MAIA_PACKAGE_PREFIX = "maia.pytree"

# List of modules to explicitly exclude from doctests, even if found and containing markers.
# Example: modules_not_to_test = ["maia.pytree.some_module_to_skip"]
modules_not_to_test = []

# --- ANSI Color Codes ---
class Colors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def cprint(color, *args, **kwargs):
  """Prints the given arguments with the specified ANSI color code."""
  print(f"{color}", end="")
  print(*args, **kwargs)
  print(Colors.ENDC, end="", flush=True) # Ensure color reset and flush output

# --- Automated Module Discovery & Filtering ---
def _scan_file_for_doctest_markers(file_path):
  """
  Reads a Python file and checks for the presence of '>>>' doctest markers.

  Args:
    file_path (str): The absolute path to the Python file.

  Returns:
    bool: True if '>>>' is found in the file content, False otherwise.
          Returns False and prints a warning if the file cannot be read.
  """
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
        return '>>>' in f.read()
  except Exception as e:
    cprint(Colors.WARNING, f"Warning: Could not read file {file_path} for doctest marker scan: {e}")
    return False

def _find_modules_with_doctest_markers(base_path, package_prefix):
  """
  Scans a directory for Python modules (.py files) that contain '>>>' doctest markers.

  It walks through the directory structure starting from `base_path`.
  Excludes __init__.py files, files/directories starting with 'test', and __pycache__ directories.
  For each valid .py file, it uses `_scan_file_for_doctest_markers` to check for '>>>'.

  Args:
      base_path (str): The root directory path to search for modules.
      package_prefix (str): The package prefix to prepend to the discovered module names
                            (e.g., "maia.pytree").

  Returns:
      list of str: A sorted list of unique, fully qualified module names that contain '>>>' markers.
  """
  modules_with_markers = []
  cprint(Colors.OKBLUE, f"Scanning for Python files with '>>>' markers in: {base_path}")
  for root, dirs, files in os.walk(base_path):
    # Prune test directories and __pycache__
    dirs[:] = [d for d in dirs if not d.lower().startswith('test') and d != '__pycache__']
    
    for file in files:
      if file.endswith(".py") and not file.startswith("__init__"):
        # Exclude test files by name pattern more robustly
        if file.lower().startswith('test_') or file.lower().endswith('_test.py'):
          continue
        
        full_file_path = os.path.join(root, file)
        if _scan_file_for_doctest_markers(full_file_path):
          relative_to_base = os.path.relpath(full_file_path, base_path)
          if relative_to_base == file: # File is directly in base_path
              module_part = relative_to_base[:-3] # Remove .py
          else:
              module_part = relative_to_base.replace(os.sep, '.')[:-3] # Convert path to module and remove .py
          
          full_module_name = f"{package_prefix}.{module_part}"
          modules_with_markers.append(full_module_name)
          cprint(Colors.OKCYAN, f"  Found '>>>' in: {full_module_name} ({full_file_path})")
        # else: # Optionally, log files without markers if needed for debugging
        #   cprint(Colors.OKCYAN, f"  No '>>>' markers in: {full_file_path}")
                  
  return sorted(list(set(modules_with_markers)))

def filter_and_confirm_doctests(module_names_with_markers):
  """
  Filters a list of module names (pre-screened for '>>>' markers) by attempting to import them
  and then using `doctest.DocTestFinder` to confirm they contain runnable doctests.
  Also respects the `modules_not_to_test` exclusion list.

  Args:
    module_names_with_markers (list of str): A list of module names that are believed
                                              to contain doctest markers ('>>>').

  Returns:
    list of str: A list of module names that were successfully imported, are not in the
                  exclusion list, and for which `DocTestFinder` found actual doctests.
  """
  confirmed_modules = []
  finder = doctest.DocTestFinder(verbose=False) # verbose=False is standard for programmatic use
  
  cprint(Colors.OKBLUE, f"\nConfirming runnable doctests in {len(module_names_with_markers)} candidate module(s)...")
  for module_name in module_names_with_markers:
    if module_name in modules_not_to_test:
      cprint(Colors.WARNING, f"Skipping module {module_name}: explicitly excluded in `modules_not_to_test`.")
      continue
    try:
      module = importlib.import_module(module_name)
      # DocTestFinder.find() returns a list of DocTest objects.
      # If this list is not empty, the module contains actual runnable doctests.
      if finder.find(module, name=module.__name__, globs=test_globs):
        confirmed_modules.append(module_name)
        cprint(Colors.OKGREEN, f"  Confirmed runnable doctests in: {module_name}")
      else:
        # This case means '>>>' was present, but DocTestFinder didn't parse any valid tests.
        cprint(Colors.OKCYAN, f"  Module {module_name} had '>>>' markers, but DocTestFinder found no runnable doctests.")
    except ImportError as e:
      cprint(Colors.WARNING, f"Warning: Could not import module {module_name} for final doctest confirmation: {e}")
    except Exception as e:
      cprint(Colors.WARNING, f"Warning: Error processing module {module_name} during final doctest confirmation: {e}")
  return confirmed_modules

# --- Doctest Globals & Hooks ---
# Global namespace for doctests, accessible by all discovered doctests.
# Includes common modules like sys, PT (maia.pytree), and np (numpy).
test_globs = {'sys': sys, 'PT': PT, 'np': np}

# Monkey-patch PT.print_tree to inject default parameters for doctest consistency.
_original_print_tree = PT.print_tree
def new_print_tree(*args, **kwargs):
  """Wrapper for PT.print_tree to set default 'out' and 'colors' for doctests."""
  kwargs.setdefault('out', sys.stdout)
  kwargs.setdefault('colors', False)
  return _original_print_tree(*args, **kwargs)
PT.print_tree = new_print_tree # Replace the original function with the new wrapper

def apply_hooks(doc_test):
    """
    Applies custom modifications to each doctest.Example within a given DocTest object.

    Current hook:
    - If an example's source code (after stripping leading whitespace) starts with "PT.new_",
      it prepends "_ = " to the stripped code, preserving original indentation.
      This handles PyTree node creation expressions that aren't assigned, ensuring they are executed.

    Args:
      doc_test (doctest.DocTest): The DocTest object whose examples are to be processed.

    Returns:
      doctest.DocTest: The (potentially) modified DocTest object.
    """
    for example in doc_test.examples:
        code = example.source
        stripped_code = code.lstrip()
        if stripped_code.startswith("PT.new_"):
            indentation = code[:len(code) - len(stripped_code)]
            example.source = f"{indentation}_ = {stripped_code}"
    return doc_test

def collect_doctests(confirmed_module_names):
  """
  Collects all `doctest.DocTest` objects from a list of confirmed module names.

  For each module name, it imports the module, finds all doctests using
  `doctest.DocTestFinder`, updates their global namespace with `test_globs`,
  and applies `apply_hooks` for any necessary example modifications.

  Args:
    confirmed_module_names (list of str): A list of fully qualified module names that
                                          are confirmed to contain runnable doctests.

  Returns:
    list of doctest.DocTest: A list of DocTest objects ready for execution.
                              Prints warnings if modules cannot be imported or processed.
  """
  tests = []
  finder = doctest.DocTestFinder(verbose=False)
  cprint(Colors.OKBLUE, f"\nCollecting DocTest objects from {len(confirmed_module_names)} module(s)...")
  for module_name in confirmed_module_names:
    try:
      module = importlib.import_module(module_name)
      # Pass module.__name__ for better reporting in DocTest objects
      for dt in finder.find(module, name=module.__name__, globs=test_globs):
        dt = apply_hooks(dt) # Apply custom hooks to each DocTest
        tests.append(dt)
      cprint(Colors.OKCYAN, f"  Collected doctests from: {module_name}")
    except ImportError as e:
      cprint(Colors.WARNING, f"CollectDoctests: Could not import module {module_name}: {e}")
    except Exception as e:
      cprint(Colors.WARNING, f"CollectDoctests: Error processing module {module_name}: {e}")
  return tests

# --- Custom Output Checkers (from original script) ---
class NoWhitespaceOutputChecker(doctest.OutputChecker):
  """
  A custom `doctest.OutputChecker` that ignores all whitespace differences
  when comparing the expected output (`want`) with the actual output (`got`).
  """
  def check_output(self, want, got, optionflags):
    normalized_want = ''.join(want.split())
    normalized_got  = ''.join(got.split())
    return normalized_want == normalized_got

WHITESPACE_PATTERNS = [
  # Patterns for which we want (yes example.want) to ignore whitespace differences
  r"DiffReport\(",
  r"CartesianCoordinates\(CoordinateX",
  r"PeriodicValues\(RotationCenter",
  r"\[\'GridLocation\',\s*array\(\[",
]

class ConditionalWhitespaceOutputChecker(doctest.OutputChecker):
  """
  A custom `doctest.OutputChecker` that ignores whitespace differences only for
  outputs matching specific regular expression patterns.
  """
  def __init__(self, patterns=None):
    super().__init__()
    self.patterns = patterns or []
  
  def check_output(self, want, got, optionflags):
    if any(re.search(pattern, want) for pattern in self.patterns):
      normalized_want = ''.join(want.split())
      normalized_got  = ''.join(got.split())
      return normalized_want == normalized_got
    else:
      return super().check_output(want, got, optionflags)

# --- Main Execution Logic ---
if __name__ == "__main__":
  cprint(Colors.HEADER, "=== Doctest Runner for Maia Modules ===")

  modules_with_markers = []
  if os.path.isdir(MAIA_PYTREE_PATH):
    modules_with_markers = _find_modules_with_doctest_markers(MAIA_PYTREE_PATH, MAIA_PACKAGE_PREFIX)
  else:
    cprint(Colors.WARNING, f"Warning: MAIA_PYTREE_PATH '{MAIA_PYTREE_PATH}' not found. Doctest collection will be skipped.")

  if not modules_with_markers:
    cprint(Colors.WARNING, "No Python modules containing '>>>' doctest markers were found. Exiting.")
    sys.exit(0)
  cprint(Colors.OKGREEN, f"Found {len(modules_with_markers)} module(s) containing '>>>' markers.")

  # Second stage: Confirm with DocTestFinder and respect exclusion list
  confirmed_runnable_modules = filter_and_confirm_doctests(modules_with_markers)

  if not confirmed_runnable_modules:
    cprint(Colors.WARNING, "No modules with runnable doctests found after DocTestFinder confirmation and exclusion. Exiting.")
    sys.exit(0)
  cprint(Colors.OKGREEN, f"{len(confirmed_runnable_modules)} module(s) confirmed to have runnable doctests.")

  # Third stage: Collect DocTest objects from confirmed modules
  collected_doctest_objects = collect_doctests(confirmed_runnable_modules)
  
  if not collected_doctest_objects:
    cprint(Colors.WARNING, "No doctest objects were collected from the confirmed modules. This is unexpected. Exiting.")
    sys.exit(0)

  cprint(Colors.OKGREEN, f"\nSuccessfully collected {len(collected_doctest_objects)} doctest object(s) to run.")
  
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
  
  cprint(Colors.HEADER, "\n=== Running Doctests ===")
  for i, doc_test_obj in enumerate(collected_doctest_objects):
    module_name_for_reporting = doc_test_obj.module.__name__ if hasattr(doc_test_obj, 'module') and hasattr(doc_test_obj.module, '__name__') else doc_test_obj.name
    cprint(Colors.OKBLUE, f"Running test {i+1}/{len(collected_doctest_objects)} from: {module_name_for_reporting} (DocTest: {doc_test_obj.name})")
    runner.run(doc_test_obj)

  cprint(Colors.HEADER, "\n=== Doctest Summary ===")
  failure_count, attempt_count = runner.summarize(verbose=True)
                                                  
  if failure_count > 0:
    cprint(Colors.FAIL, f"\n{failure_count} out of {attempt_count} doctests FAILED.")
    sys.exit(1)
  else:
    cprint(Colors.OKGREEN, f"\nAll {attempt_count} doctests PASSED.")
    sys.exit(0)
