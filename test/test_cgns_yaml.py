import pytest
import os
import tempfile

from maia.utils.yaml import parse_cgns_yaml
from maia.utils.yaml import parse_yaml_cgns

from maia       import pytree     as PT
from maia.utils import test_utils as TU

def test_yaml_loading():
  """ In order to facilitate the versionning of the data and results files used
  for unit tests and basic functional tests, maia works with small yaml files 
  describing CGNS meshes.

  Yaml files can easy be converted into standard PyCGNS nodes or trees using the yaml parser
  """

  # Here, the yaml file is converted into a complete cgns tree
  filename = os.path.join(TU.mesh_dir,'S_twoblocks.yaml')
  with open(filename, 'r') as f:
    tree = parse_yaml_cgns.to_cgns_tree(f)

  assert tree[3] == 'CGNSTree_t'

  # Sometimes, files contains only a lower level cgns node, 
  # eg a FlowSolution containing a reference value:
  yt = """\
FlowSolution FlowSolution_t:
  GridLocation GridLocation_t "Vertex":
  Array1 DataArray_t R4 [1., 2., 3., 4.]:
  Array2 DataArray_t R4 [-10., -20., -30., -40.]:
"""
  # In this case, the parser can load this node using
  node = parse_yaml_cgns.to_node(yt)
  assert node[3] == 'FlowSolution_t' and len(node[2]) == 3

  # Note that the yaml parser works on stream : loading from a file or a string is equivalent
  with tempfile.TemporaryFile(mode='w+') as f:
    f.write(yt)
    f.seek(0)
    node_from_tree = parse_yaml_cgns.to_node(f)
  assert PT.is_same_tree(node_from_tree, node)

@pytest.mark.parametrize("filename", ["U_ATB_45.yaml", "S_twoblocks.yaml"])
def test_cgns_yaml_conversion(filename):
  """ CGNS meshes can also be converted into yaml files, eg in order to store references
  solutions on the disk.
  """

  with open(os.path.join(TU.mesh_dir,filename), 'r') as f:
    input_yaml_lines = f.readlines()
  input_yaml_lines = [line for line in input_yaml_lines if line[0] != '#'] #Filter comments in file
  input_yaml = ''.join(input_yaml_lines)

  tree = parse_yaml_cgns.to_cgns_tree(input_yaml)

  #Convert the CGNS tree to a yaml lines
  yaml = parse_cgns_yaml.to_yaml(tree)
  yaml = '\n'.join(yaml) #yaml is now a huge string

  for input_line, new_line in zip(input_yaml_lines, yaml.split('\n')):
    assert input_line == new_line + '\n' #Readlines adds a '\n' at the end of each lines

  # Yaml can be written in file
  with tempfile.TemporaryFile(mode='w+') as f:
    f.write(yaml)
    #Re read file for check
    f.seek(0)
    new_tree = parse_yaml_cgns.to_cgns_tree(f)

  assert PT.is_same_tree(tree, new_tree)
