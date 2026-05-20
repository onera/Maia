import pytest
from mpi4py import MPI

def is_master_process(config):
  if config.getoption('scheduler', None) == 'dynamic':
    from pytest_parallel.utils.mpi import is_dyn_master_process
    is_master = is_dyn_master_process(MPI.COMM_WORLD)
  else:
    is_master = MPI.COMM_WORLD.Get_rank() == 0
  return is_master


def rewrite_junit_report(xmlpath, suite_name):
  from pathlib import Path
  from collections import defaultdict
  import xml.etree.ElementTree as ET

  if not xmlpath or not Path(xmlpath).exists():
    return

  tree = ET.parse(xmlpath)
  root = tree.getroot()

  counts = defaultdict(int)

  for testcase in root.iter("testcase"):

    parts = testcase.get('classname', '').split('.')
    if parts[-1].startswith('Test'): # Class embedded test
        classname = parts.pop() + '.'
    else:
        classname = ''

    new_name = classname + testcase.get("name")

    # Gitlab require unique test names, add a suffix if needed
    counts[new_name] += 1
    if (c := counts[new_name]) > 1:
        new_name += f'.{c}'

    testcase.set("classname", suite_name)
    testcase.set("name", new_name)
    testcase.set("file", '/'.join(parts) + '.py')

    # ET.indent(tree) Require Python >= 3.9
    tree.write(xmlpath, encoding="utf-8", xml_declaration=True)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
  #Only master process holds test results, others are empty
  if is_master_process(config):
    config.option.xmlpath = "junit_maia_unit.xml"
    
def pytest_sessionfinish(session, exitstatus):
  if is_master_process(session.config):
    rewrite_junit_report(session.config.option.xmlpath, 'Unitary')