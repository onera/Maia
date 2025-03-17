import os
import pytest
from maia.utils.script_utils import determine_output_file_path

def test_determine_output_path_none():
  input_path = "/some/path/data.hdf"
  output_default_extension = ".txt"
  result = determine_output_file_path(input_path, None, output_default_extension)
  assert result == "data.txt"

def test_determine_output_path_directory(tmp_path):
  input_path = "data.hdf"
  output_default_extension = ".txt"
  output_dir = tmp_path / "output_dir"
  output_dir.mkdir()
  result = determine_output_file_path(input_path, str(output_dir), output_default_extension)
  expected = os.path.join(str(output_dir), "data" + output_default_extension)
  assert result == expected

def test_determine_output_path_file():
  input_path = "data.hdf"
  output_default_extension = ".txt"
  output_file = "custom_output.txt"
  result = determine_output_file_path(input_path, output_file, output_default_extension)
  assert result == output_file

