import pytest_parallel

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns

from cmaia.part_algo                   import cgns_registry as CGR
from maia.pytree.cgns_keywords         import Label as CGL
from maia.algo.part.cgns_registry.tree import make_cgns_registry, add_cgns_registry_information


yt_1p = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
    ZoneBC ZoneBC_t:
      WALL BC_t:
      FARFIELD BC_t:
  ZoneU2 Zone_t [[216,125,0]]:
    ZoneBC ZoneBC_t:
      WALL BC_t:
      SYM BC_t:
  WALL Family_t:
  FARFIELD Family_t:
  SYM Family_t:
"""

yt_2p = ["""
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
    ZoneBC ZoneBC_t:
      WALL BC_t:
      FARFIELD BC_t:
  WALL Family_t:
  FARFIELD Family_t:
  SYM Family_t:
""",
"""
Base0 CGNSBase_t [3,3]:
  ZoneU2 Zone_t [[216,125,0]]:
    ZoneBC ZoneBC_t:
      WALL BC_t:
      SYM BC_t:
  WALL Family_t:
  FARFIELD Family_t:
  SYM Family_t:
"""]


@pytest_parallel.mark.parallel(1)
def test_cgns_registry_1p(comm):
  """
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt_1p)

  cgr = make_cgns_registry(tree, comm)

  assert list(cgr.paths(CGL.Zone_t)) == ["/Base0/ZoneU1", "/Base0/ZoneU2"]

  # get_global_id_from_path_and_type
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0", CGL.CGNSBase_t) == 1

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", CGL.Zone_t) == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", CGL.Zone_t) == 2

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", CGL.BC_t) == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , CGL.BC_t) == 2

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , CGL.BC_t) == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", CGL.BC_t) == 4

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , CGL.Family_t) == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", CGL.Family_t) == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , CGL.Family_t) == 2

  # get_global_id_from_path_and_type
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0", "CGNSBase_t") == 1

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", "Zone_t") == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", "Zone_t") == 2

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", "BC_t") == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , "BC_t") == 2

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , "BC_t") == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", "BC_t") == 4

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , "Family_t") == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", "Family_t") == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , "Family_t") == 2

  # get_path_from_global_id_and_type
  assert CGR.get_path_from_global_id_and_type(cgr, 1,  CGL.CGNSBase_t) == "/Base0"

  assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.Zone_t) == "/Base0/ZoneU1"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.Zone_t) == "/Base0/ZoneU2"

  assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.BC_t) == "/Base0/ZoneU1/ZoneBC/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.BC_t) == "/Base0/ZoneU1/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, CGL.BC_t) == "/Base0/ZoneU2/ZoneBC/SYM"
  assert CGR.get_path_from_global_id_and_type(cgr, 4, CGL.BC_t) == "/Base0/ZoneU2/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, CGL.Family_t) == "/Base0/WALL"
  assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.Family_t) == "/Base0/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.Family_t) == "/Base0/SYM"

  # get_path_from_global_id_and_type
  assert CGR.get_path_from_global_id_and_type(cgr, 1,  "CGNSBase_t") == "/Base0"

  assert CGR.get_path_from_global_id_and_type(cgr, 1, "Zone_t") == "/Base0/ZoneU1"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, "Zone_t") == "/Base0/ZoneU2"

  assert CGR.get_path_from_global_id_and_type(cgr, 1, "BC_t") == "/Base0/ZoneU1/ZoneBC/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, "BC_t") == "/Base0/ZoneU1/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, "BC_t") == "/Base0/ZoneU2/ZoneBC/SYM"
  assert CGR.get_path_from_global_id_and_type(cgr, 4, "BC_t") == "/Base0/ZoneU2/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, "Family_t") == "/Base0/WALL"
  assert CGR.get_path_from_global_id_and_type(cgr, 1, "Family_t") == "/Base0/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, "Family_t") == "/Base0/SYM"

@pytest_parallel.mark.parallel(2)
def test_cgns_registry_2p(comm):
  """
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt_2p[comm.Get_rank()])
  cgr = make_cgns_registry(tree, comm)

  if comm.Get_rank()==0:
    assert list(cgr.paths(CGL.Zone_t)) == ["/Base0/ZoneU1"]
  if comm.Get_rank()==1:
    assert list(cgr.paths(CGL.Zone_t)) == ["/Base0/ZoneU2"]

  # get_global_id_from_path_and_type
  if comm.Get_rank()==0:
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", CGL.Zone_t) == 1
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", CGL.BC_t) == 1
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , CGL.BC_t) == 2
  if comm.Get_rank()==1:
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", CGL.Zone_t) == 2
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", CGL.Zone_t) == 2
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , CGL.BC_t) == 3
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", CGL.BC_t) == 4

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , CGL.Family_t) == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", CGL.Family_t) == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , CGL.Family_t) == 2

  # get_global_id_from_path_and_type
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0", "CGNSBase_t") == 1

  if comm.Get_rank()==0:
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", "Zone_t") == 1
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", "BC_t") == 1
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , "BC_t") == 2
  if comm.Get_rank()==1:
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", "Zone_t") == 2
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , "BC_t") == 3
    assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", "BC_t") == 4

  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , "Family_t") == 3
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", "Family_t") == 1
  assert CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , "Family_t") == 2

  # get_path_from_global_id_and_type
  assert CGR.get_path_from_global_id_and_type(cgr, 1,  CGL.CGNSBase_t) == "/Base0"

  if comm.Get_rank()==0:
    assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.Zone_t) == "/Base0/ZoneU1"
    assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.BC_t) == "/Base0/ZoneU1/ZoneBC/FARFIELD"
    assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.BC_t) == "/Base0/ZoneU1/ZoneBC/WALL"
  if comm.Get_rank()==1:
    assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.Zone_t) == "/Base0/ZoneU2"
    assert CGR.get_path_from_global_id_and_type(cgr, 3, CGL.BC_t) == "/Base0/ZoneU2/ZoneBC/SYM"
    assert CGR.get_path_from_global_id_and_type(cgr, 4, CGL.BC_t) == "/Base0/ZoneU2/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, CGL.Family_t) == "/Base0/WALL"
  assert CGR.get_path_from_global_id_and_type(cgr, 1, CGL.Family_t) == "/Base0/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, CGL.Family_t) == "/Base0/SYM"

  # get_path_from_global_id_and_type
  assert CGR.get_path_from_global_id_and_type(cgr, 1,  "CGNSBase_t") == "/Base0"

  if comm.Get_rank()==0:
    assert CGR.get_path_from_global_id_and_type(cgr, 1, "Zone_t") == "/Base0/ZoneU1"
    assert CGR.get_path_from_global_id_and_type(cgr, 1, "BC_t") == "/Base0/ZoneU1/ZoneBC/FARFIELD"
    assert CGR.get_path_from_global_id_and_type(cgr, 2, "BC_t") == "/Base0/ZoneU1/ZoneBC/WALL"
  if comm.Get_rank()==1:
    assert CGR.get_path_from_global_id_and_type(cgr, 2, "Zone_t") == "/Base0/ZoneU2"
    assert CGR.get_path_from_global_id_and_type(cgr, 3, "BC_t") == "/Base0/ZoneU2/ZoneBC/SYM"
    assert CGR.get_path_from_global_id_and_type(cgr, 4, "BC_t") == "/Base0/ZoneU2/ZoneBC/WALL"

  assert CGR.get_path_from_global_id_and_type(cgr, 3, "Family_t") == "/Base0/WALL"
  assert CGR.get_path_from_global_id_and_type(cgr, 1, "Family_t") == "/Base0/FARFIELD"
  assert CGR.get_path_from_global_id_and_type(cgr, 2, "Family_t") == "/Base0/SYM"

@pytest_parallel.mark.parallel(1)
def test_add_cgns_registry_information_1p(comm):
  """
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt_1p)
  cgr = add_cgns_registry_information(tree, comm)

  zone1_id_n = PT.get_node_from_path(tree, "Base0/ZoneU1/:CGNS#Registry")
  zone2_id_n = PT.get_node_from_path(tree, "Base0/ZoneU2/:CGNS#Registry")

  assert PT.get_value(zone1_id_n) == 1
  assert PT.get_value(zone2_id_n) == 2

@pytest_parallel.mark.parallel(2)
def test_add_cgns_registry_information_2p(comm):
  """
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt_2p[comm.Get_rank()])
  cgr = add_cgns_registry_information(tree, comm)

  if comm.Get_rank()==0:
    assert PT.get_value(PT.get_node_from_path(tree, "Base0/ZoneU1/:CGNS#Registry")) == 1
  if comm.Get_rank()==1:
    assert PT.get_value(PT.get_node_from_path(tree, "Base0/ZoneU2/:CGNS#Registry")) == 2

