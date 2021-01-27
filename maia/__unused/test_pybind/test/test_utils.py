
import pytest
import pytest_check as check

# @pytest.mark.parametrize("make_sub_comm", [1], indirect=['make_sub_comm'])
# --------------------------------------------------------------------------
@pytest.mark.mpi_test(comm_size=1)
def test_utils(make_sub_comm):
  """
  """
  # print("make_sub_comm :: ", make_sub_comm)
  print(f"[{make_sub_comm.Get_rank()}/{make_sub_comm.Get_size()}] test_utils")
  # from maia.utils import dispatch as MUD
  # print("test_utils", dir(MUD))
  assert 0 == 0
  # check.equal(0, 0, " hehe1" )
  # check.equal(0, 1, " hehe2" )
  # check.equal(0, 1, " hehe3" )

# --------------------------------------------------------------------------
@pytest.mark.mpi_test(comm_size=1)
@pytest.mark.parametrize("make_sub_comm", [1, 2], indirect=['make_sub_comm'])
def test_utils2(make_sub_comm):
  """
  """
  # print("make_sub_comm :: ", make_sub_comm)
  print(f"[{make_sub_comm.Get_rank()}/{make_sub_comm.Get_size()}] test_utils2")
  # print(f"test_utils2 on rank {}")
  assert 0 == 0
  # check.equal(0, 1, " hehe4" )
  # check.equal(0, 1, " hehe5" )
  # check.equal(0, 0, " hehe6" )

# # --------------------------------------------------------------------------
# @pytest.mark.mpi_test(comm_size=2)
# def test_utils3(make_sub_comm):
#   """
#   """

# # --------------------------------------------------------------------------
# @pytest.mark.mpi_test(comm_size=1)
# def test_utils4(make_sub_comm):
#   """
#   """

  # from maia.utils import dispatch as MUD
  # print("test_utils2",  dir(MUD.kind))
  # print(MUD.kind.CGNSTree_t)
  # print(MUD.kind.CGNSBase_t)
  # print(MUD.kind.Zone_t)
  # print("******", MUD.kind.Zone_t.name, dir(MUD.kind.Zone_t.name))
  # print(type(MUD.kind))
  # MUD.test_enum(MUD.kind.CGNSTree_t)
  # MUD.test_enum(MUD.kind.CGNSBase_t)
  # MUD.test_enum(MUD.kind.Zone_t)
# --------------------------------------------------------------------------
# @pytest.mark.mpi_test(comm_size=2)
# def test_utils3():
#   """
#   """
#   print("test_utils3")
#   assert 0 == 0
#   check.equal(0, 1, " hehe4" )
#   check.equal(0, 1, " hehe5" )
#   check.equal(0, 0, " hehe6" )

# --------------------------------------------------------------------------
# def test_first_step():
#   """
#   """
#   # print("test_first_step")
#   from maia.utils import boundary_algorihms as MUB
#   # print(MUB.__file__)
#   from maia.utils import first_step as MUF

#   base = MUF.cgns_base("cgns_base", 1)

#   zone_u1 = MUF.zone_unstructured("zone_name", 1)
#   assert(zone_u1.global_id == 1          );
#   assert(zone_u1.name      == "zone_name");

#   zone_s1 = MUF.zone_structured("cartesian", 2)
#   assert(zone_s1.global_id == 2          );
#   assert(zone_s1.name      == "cartesian");

#   # MUF.add_zone_to_base(base, zone_u1);
#   # MUF.add_zone_to_base(base, zone_s1);

#   print(zone_u1.global_id)
#   print(zone_s1.global_id)
#   print(base)


# def test_automatic_dispatch():
#   """
#   """
#   from maia.utils import boundary_algorihms as MUB
#   import numpy as NPY

#   face_vtx_32  = NPY.empty(10, dtype='int32', order='F')
#   face_vtx_64  = NPY.empty(10, dtype='int64', order='F')
#   face_vtx_idx = NPY.empty(3 , dtype='int32', order='F')

#   face_vtx_u64  = NPY.empty(10, dtype='uint64', order='F')

#   MUB.automatic_dispatch(face_vtx_32, face_vtx_idx)
#   MUB.automatic_dispatch(face_vtx_64, face_vtx_idx)

#   try:
#     MUB.automatic_dispatch(face_vtx_u64, face_vtx_idx)
#   except TypeError:
#     pass

# def test_automatic_dispatch_pybind():
#   """
#   """
#   from maia.utils import dispatch as MUD
#   import numpy as NPY

#   face_vtx_32  = NPY.empty(10, dtype='int32', order='F')
#   face_vtx_64  = NPY.empty(10, dtype='int64', order='F')
#   face_vtx_idx = NPY.empty(3 , dtype='int32', order='F')

#   face_vtx_u64  = NPY.empty(10, dtype='double', order='F')

#   MUD.auto_dispatch(face_vtx_32, face_vtx_idx)
#   MUD.auto_dispatch(face_vtx_64, face_vtx_idx)

#   # print(help(MUD.auto_dispatch))

#   try:
#     MUD.auto_dispatch(face_vtx_u64, face_vtx_idx)
#   except TypeError:
#     pass
