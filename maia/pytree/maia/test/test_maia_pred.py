import maia.pytree as PT

import maia.pytree.maia.pred as MTp

def test_predicates():
  zone = PT.yaml.to_node("""
  Zone Zone_t:
    FSCC FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Pressure DataArray_t:
    EmptyDDVtx DiscreteData_t:
    PartialDDCC DiscreteData_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t "CellCenter":
      Pressure DataArray_t:
  """)

  assert [PT.get_name(n) for n in PT.get_children_from_predicate(zone, MTp.FULL_CTN_CELL)] == ['FSCC']
  assert [PT.get_name(n) for n in PT.get_children_from_predicate(zone, MTp.FULL_CTN_VTX)] == []