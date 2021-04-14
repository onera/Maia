from pytest_mpi_check            import assert_mpi
from pytest_mpi_check._decorator import mark_mpi_test
from maia.utils import parse_yaml_cgns
import Converter.Internal as     I

import cmaia.cgns_registry.cgns_registry     as     CGR
from maia.cgns_registry.tree                 import make_cgns_registry, add_cgns_registry_information
from maia.cgns_registry import cgns_keywords as     CGK


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


@mark_mpi_test(1)
def test_cgns_registry_1p(sub_comm):
  """
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt_1p)

  cgr = make_cgns_registry(tree, sub_comm)

  assert( list(cgr.paths(CGK.kind.Zone_t)) == ["/Base0/ZoneU1", "/Base0/ZoneU2"])

  # get_global_id_from_path_and_type
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0", CGK.kind.CGNSBase_t) == 1)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", CGK.kind.Zone_t) == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", CGK.kind.Zone_t) == 2)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", CGK.kind.BC_t) == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , CGK.kind.BC_t) == 2)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , CGK.kind.BC_t) == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", CGK.kind.BC_t) == 4)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , CGK.kind.Family_t) == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", CGK.kind.Family_t) == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , CGK.kind.Family_t) == 2)

  # get_global_id_from_path_and_type
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0", "CGNSBase_t") == 1)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", "Zone_t") == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", "Zone_t") == 2)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", "BC_t") == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , "BC_t") == 2)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , "BC_t") == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", "BC_t") == 4)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , "Family_t") == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", "Family_t") == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , "Family_t") == 2)

  # get_path_from_global_id_and_type
  assert(CGR.get_path_from_global_id_and_type(cgr, 1,  CGK.kind.CGNSBase_t) == "/Base0")

  assert(CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.Zone_t) == "/Base0/ZoneU1")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.Zone_t) == "/Base0/ZoneU2")

  assert(CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.BC_t) == "/Base0/ZoneU1/ZoneBC/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.BC_t) == "/Base0/ZoneU1/ZoneBC/WALL"    )

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, CGK.kind.BC_t) == "/Base0/ZoneU2/ZoneBC/SYM" )
  assert(CGR.get_path_from_global_id_and_type(cgr, 4, CGK.kind.BC_t) == "/Base0/ZoneU2/ZoneBC/WALL")

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, CGK.kind.Family_t) == "/Base0/WALL"    )
  assert(CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.Family_t) == "/Base0/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.Family_t) == "/Base0/SYM"     )

  # get_path_from_global_id_and_type
  assert(CGR.get_path_from_global_id_and_type(cgr, 1,  "CGNSBase_t") == "/Base0")

  assert(CGR.get_path_from_global_id_and_type(cgr, 1, "Zone_t") == "/Base0/ZoneU1")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, "Zone_t") == "/Base0/ZoneU2")

  assert(CGR.get_path_from_global_id_and_type(cgr, 1, "BC_t") == "/Base0/ZoneU1/ZoneBC/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, "BC_t") == "/Base0/ZoneU1/ZoneBC/WALL"    )

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, "BC_t") == "/Base0/ZoneU2/ZoneBC/SYM" )
  assert(CGR.get_path_from_global_id_and_type(cgr, 4, "BC_t") == "/Base0/ZoneU2/ZoneBC/WALL")

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, "Family_t") == "/Base0/WALL"    )
  assert(CGR.get_path_from_global_id_and_type(cgr, 1, "Family_t") == "/Base0/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, "Family_t") == "/Base0/SYM"     )

@mark_mpi_test(2)
def test_cgns_registry_2p(sub_comm):
  """
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt_2p[sub_comm.Get_rank()])

  cgr = make_cgns_registry(tree, sub_comm)

  assert_mpi(sub_comm, 0, list(cgr.paths(CGK.kind.Zone_t)) == ["/Base0/ZoneU1"])
  assert_mpi(sub_comm, 1, list(cgr.paths(CGK.kind.Zone_t)) == ["/Base0/ZoneU2"])

  # get_global_id_from_path_and_type
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0", CGK.kind.CGNSBase_t) == 1)

  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", CGK.kind.Zone_t) == 1)
  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", CGK.kind.Zone_t) == 2)

  assert_mpi(sub_comm, 1, lambda x: CGR.get_global_id_from_path_and_type(cgr, x, CGK.kind.Zone_t) == 2, "/Base0/ZoneU2")

  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", CGK.kind.BC_t) == 1)
  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , CGK.kind.BC_t) == 2)

  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , CGK.kind.BC_t) == 3)
  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", CGK.kind.BC_t) == 4)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , CGK.kind.Family_t) == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", CGK.kind.Family_t) == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , CGK.kind.Family_t) == 2)

  # get_global_id_from_path_and_type
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0", "CGNSBase_t") == 1)

  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1", "Zone_t") == 1)
  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2", "Zone_t") == 2)

  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/FARFIELD", "BC_t") == 1)
  assert_mpi(sub_comm, 0, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU1/ZoneBC/WALL"    , "BC_t") == 2)

  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/SYM" , "BC_t") == 3)
  assert_mpi(sub_comm, 1, lambda: CGR.get_global_id_from_path_and_type(cgr, "/Base0/ZoneU2/ZoneBC/WALL", "BC_t") == 4)

  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/WALL"    , "Family_t") == 3)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/FARFIELD", "Family_t") == 1)
  assert(CGR.get_global_id_from_path_and_type(cgr, "/Base0/SYM"     , "Family_t") == 2)

  # get_path_from_global_id_and_type
  assert(CGR.get_path_from_global_id_and_type(cgr, 1,  CGK.kind.CGNSBase_t) == "/Base0")

  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.Zone_t) == "/Base0/ZoneU1")
  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.Zone_t) == "/Base0/ZoneU2")

  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.BC_t) == "/Base0/ZoneU1/ZoneBC/FARFIELD")
  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.BC_t) == "/Base0/ZoneU1/ZoneBC/WALL"    )

  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 3, CGK.kind.BC_t) == "/Base0/ZoneU2/ZoneBC/SYM" )
  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 4, CGK.kind.BC_t) == "/Base0/ZoneU2/ZoneBC/WALL")

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, CGK.kind.Family_t) == "/Base0/WALL"    )
  assert(CGR.get_path_from_global_id_and_type(cgr, 1, CGK.kind.Family_t) == "/Base0/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, CGK.kind.Family_t) == "/Base0/SYM"     )

  # get_path_from_global_id_and_type
  assert(CGR.get_path_from_global_id_and_type(cgr, 1,  "CGNSBase_t") == "/Base0")

  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 1, "Zone_t") == "/Base0/ZoneU1")
  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 2, "Zone_t") == "/Base0/ZoneU2")

  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 1, "BC_t") == "/Base0/ZoneU1/ZoneBC/FARFIELD")
  assert_mpi(sub_comm, 0, lambda: CGR.get_path_from_global_id_and_type(cgr, 2, "BC_t") == "/Base0/ZoneU1/ZoneBC/WALL"    )

  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 3, "BC_t") == "/Base0/ZoneU2/ZoneBC/SYM" )
  assert_mpi(sub_comm, 1, lambda: CGR.get_path_from_global_id_and_type(cgr, 4, "BC_t") == "/Base0/ZoneU2/ZoneBC/WALL")

  assert(CGR.get_path_from_global_id_and_type(cgr, 3, "Family_t") == "/Base0/WALL"    )
  assert(CGR.get_path_from_global_id_and_type(cgr, 1, "Family_t") == "/Base0/FARFIELD")
  assert(CGR.get_path_from_global_id_and_type(cgr, 2, "Family_t") == "/Base0/SYM"     )

@mark_mpi_test(1)
def test_add_cgns_registry_information_1p(sub_comm):
  """
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt_1p)
  cgr = add_cgns_registry_information(tree, sub_comm)

  zone1_id_n = I.getNodeFromPath(tree, "/Base0/ZoneU1/:CGNS#Registry")
  zone2_id_n = I.getNodeFromPath(tree, "/Base0/ZoneU2/:CGNS#Registry")

  assert(I.getValue(zone1_id_n) == 1)
  assert(I.getValue(zone2_id_n) == 2)

@mark_mpi_test(2)
def test_add_cgns_registry_information_2p(sub_comm):
  """
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt_2p[sub_comm.Get_rank()])
  cgr = add_cgns_registry_information(tree, sub_comm)

  assert_mpi(sub_comm, 0, lambda: I.getValue(I.getNodeFromPath(tree, "/Base0/ZoneU1/:CGNS#Registry")) == 1)
  assert_mpi(sub_comm, 1, lambda: I.getValue(I.getNodeFromPath(tree, "/Base0/ZoneU2/:CGNS#Registry")) == 2)

