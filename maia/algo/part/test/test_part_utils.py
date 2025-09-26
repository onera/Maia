import pytest_parallel

import maia.pytree as PT

from maia.algo.part import utils

@pytest_parallel.mark.parallel(3)
def test_all_containers(comm):
  if comm.rank == 0:
    zones = PT.yaml.to_nodes("""
    Zone.P0.N0 Zone_t:
      FS FlowSolution_t:
      ZSR ZoneSubRegion_t:
    """)
  elif comm.rank == 1:
    zones = PT.yaml.to_nodes("""
    Zone.P1.N0 Zone_t:
      FS FlowSolution_t:
      ZSR ZoneSubRegion_t:
    Zone.P1.N1 Zone_t:
      FS FlowSolution_t:
      OtherFS FlowSolution_t:
      ZSR ZoneSubRegion_t:
    """)
  else:
    zones = []

  assert utils.gather_containers_name(zones, PT.pred.label_is('FlowSolution_t'), 'all', comm) == ['FS']
  assert utils.gather_containers_name(zones, PT.pred.label_is('FlowSolution_t'), 'any', comm) == ['FS', 'OtherFS']