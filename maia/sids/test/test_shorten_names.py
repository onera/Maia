import pytest

from maia.sids.shorten_names import shorten_names,shorten_field_names
from maia.utils import parse_yaml_cgns

class Test_shorten_names:
  yt = """
  MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
    TurbulentSANuTildeDensityGradientX DataArray_t:
    TurbulentSANuTildeDensityGradientY DataArray_t:
    MyShortName DataArray_t:
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt)

  def test_shorten_field_names(self):
      shorten_field_names(self.tree,quiet=True)

      expected_yt = """
      MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_tree = parse_yaml_cgns.to_complete_pytree(expected_yt)
      assert self.tree == expected_tree

  def test_shorten_names(self):
      shorten_names(self.tree,quiet=True)

      expected_yt = """
      MyVeryLoooFlowSoluName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_tree = parse_yaml_cgns.to_complete_pytree(expected_yt)
      print(self.tree)
      assert self.tree == expected_tree
