import mpi4py
import pytest
import h5py
import subprocess
import re
import os

import maia
import maia.pytree as PT

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
RED = '\x1b[91m'
PURPLE = '\x1b[95m'
ENDC = '\x1b[0m'
YELLOW = '\033[93m'
CYAN = '\033[36m'

vendor, version = mpi4py.MPI.get_vendor()
MPI_ENV_VARS = ["MPI_", "PMI_", "PMI_", "PMIX_", "I_MPI_"]
if vendor == 'Open MPI' and version[0] == 4:
  MPI_ENV_VARS.append('OMPI_')

def subprocess_run(*args, **kwargs):
  """ If tests are launched with mpirun, we have a strange crash
  because maia_cgns_check try to initialize MPI
  This function remove MPI env variables to simulate a sequential execution
  """
  env = os.environ.copy()
  for var in list(env.keys()):
    if any(var.startswith(s) for s in MPI_ENV_VARS):
      del env[var]
  env['MAIA_DISABLE_BETA_MSG'] = "1"
  return subprocess.run(*args, **kwargs, env=env)

def strip_ansi_sequences(text):
  """
  Removes ANSI escape sequences from a given string.
  """
  return ANSI_ESCAPE.sub('', text)

def test_usage():
  out = subprocess_run(['maia_cgns_check'], capture_output=True)
  assert out.returncode != 0
  assert b'the following arguments are required: cgns_file' in out.stderr

  # Can not use explain with filename
  out = subprocess_run(['maia_cgns_check', 'out.cgns', '--explain', 'E212'], capture_output=True)
  assert out.returncode != 0
  assert b'maia_cgns_check: error: unrecognized arguments: out.cgns' in out.stderr


def test_explain():
  out = subprocess_run(['maia_cgns_check', '--explain', 'E212'], capture_output=True)
  assert out.stdout.startswith(b'E212 - Invalid label\n\n')

  # Code 'E' can be ommit
  out = subprocess_run(['maia_cgns_check', '--explain', '212'], capture_output=True)
  assert out.stdout.startswith(b'E212 - Invalid label\n\n')

  out = subprocess_run(['maia_cgns_check', '--explain', 'E512'], capture_output=True)
  assert out.stdout == b'Rule E512 is not a valid rule\n'

def test_fatal(tmp_path):
  out = subprocess_run(['maia_cgns_check', tmp_path / 'missing.cgns'], capture_output=True)
  assert out.returncode != 0
  assert f'Execution aborted due to {RED}fatal error' in out.stdout.decode()


def test_stage1_links(tmp_path):
  link_dir = tmp_path / 'FIELDS'
  link_dir.mkdir()

  # File storing links
  ftree = PT.yaml.to_node("""
  CGNSTree CGNSTree_t:
    TempFields UserDefinedData_t:
      Fields@1 FlowSolution_t:
        Pressure DataArray_t R8 [1,2,3,4]:
      Fields@2 FlowSolution_t:
        Pressure DataArray_t R8 [11,12,13,14]:
      Fields@3 FlowSolution_t:
        Pressure DataArray_t R8 [21,22,23,24]:
  """)
  maia.io.write_tree(ftree, link_dir / 'fields.cgns')
  # Create a trouble in linked file
  with h5py.File(link_dir / 'fields.cgns', 'r+') as f:
    f['/TempFields/Fields@2'].attrs['unexpected'] = 'wrong'

  # Main file
  tree = PT.yaml.to_cgns_tree("""
  Base CGNSBase_t [2,2]:
    Zone Zone_t [[4, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      QUAD Elements_t [7, 0]:
        ElementRange IndexRange_t [1, 1]:
        ElementConnectivity DataArray_t [1, 2, 3, 4]:
      FlowSolution FlowSolution_t:
  """)
  maia.io.write_tree(tree, tmp_path / 'main.cgns', links=[['', './FIELDS/fields.cgns', 'TempFields/Fields@2', 'Base/Zone/FlowSolution']])

  out = subprocess_run(['maia_cgns_check', tmp_path/'main.cgns'], capture_output=True)
  assert out.stdout.decode() == \
    f"/Base/Zone/FlowSolution {CYAN}(-> FIELDS/fields.cgns::/TempFields/Fields@2){ENDC}: {YELLOW}W115{ENDC} Unexpected HDF5 attributes: {{'unexpected'}} \n"

def test_simple_check(tmp_path):
  tree = PT.yaml.to_cgns_tree("""
  Zone_t Zone_t:
    ZoneBC ZoneBC_t:
      BC BC_t 'Null':
        GridLocation GridLocation_t 'FaceCenter':
  """)

  filepath = tmp_path / 'test.cgns'
  maia.io.write_tree(tree, filepath)

  out = subprocess_run(['maia_cgns_check', filepath], capture_output=True)
  assert out.stdout.decode() == f"""\
/Base/Zone_t: {RED}E214{ENDC} Missing value for Zone_t node, which should be of kind I
/Base/Zone_t: {RED}E226{ENDC} Missing required child of label ZoneType_t
/Base/Zone_t: {PURPLE}Unable to check E233,E216,E231,E235 due to other errors{ENDC}
/Base/Zone_t/ZoneBC/BC: {RED}E227{ENDC} Exactly one child among ('PointList', 'PointRange') is required, but none were found
/Base/Zone_t/ZoneBC/BC/GridLocation: {PURPLE}Unable to check E241 due to other errors{ENDC}
"""


def test_ignore(tmp_path):
  tree = PT.new_CGNSTree()
  PT.new_Descriptor('WrongDescr', 'Should not be here', parent=tree)
  
  filepath = tmp_path / 'test.cgns'
  maia.io.write_tree(tree, filepath)

  out = subprocess_run(['maia_cgns_check', filepath], capture_output=True)
  assert b'Child of label Descriptor_t is not allowed under a CGNSTree_t parent' in out.stdout

  # Error is ignored
  out = subprocess_run(['maia_cgns_check', filepath, '--ignore', 'E222'], capture_output=True)
  assert out.stdout == b''
