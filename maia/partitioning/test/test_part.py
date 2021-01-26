import numpy as np
from   maia.utils        import parse_yaml_cgns
import maia.partitioning.part as part


def test_get_matching_joins():
  yt = """
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchI GridConnectivity_t "ZoneC":
        Ordinal UserDefinedData_t 1:
        OrdinalOpp UserDefinedData_t 3:
      matchII GridConnectivity_t "ZoneB":
        Ordinal UserDefinedData_t 2:
        OrdinalOpp UserDefinedData_t 4:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchIII GridConnectivity_t "ZoneA":
        Ordinal UserDefinedData_t 4:
        OrdinalOpp UserDefinedData_t 2:
  ZoneC Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchIV GridConnectivity_t "ZoneA":
        Ordinal UserDefinedData_t 3:
        OrdinalOpp UserDefinedData_t 1:
"""
  dist_tree = parse_yaml_cgns.to_complete_pytree(yt)
  assert (part.get_matching_joins(dist_tree) == np.array([3,4,1,2])-1).all()
