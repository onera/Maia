
# --------------------------------------------------------------------------
def test_utils():
  """
  """
  print("test_first_step")
  assert 0 == 0

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


def test_automatic_dispatch():
  """
  """
  from maia.utils import boundary_algorihms as MUB
  import numpy as NPY
  print(MUB.__file__)

  face_vtx_32  = NPY.empty(10, dtype='int32', order='F')
  face_vtx_64  = NPY.empty(10, dtype='int64', order='F')
  face_vtx_idx = NPY.empty(3 , dtype='int32', order='F')

  face_vtx_u64  = NPY.empty(10, dtype='uint64', order='F')

  MUB.automatic_dispatch(face_vtx_32, face_vtx_idx)
  MUB.automatic_dispatch(face_vtx_64, face_vtx_idx)

  try:
    MUB.automatic_dispatch(face_vtx_u64, face_vtx_idx)
  except TypeError:
    print("Bad type pass")
    pass

def test_automatic_dispatch_pybind():
  """
  """
  from maia.utils import dispatch as MUD
  import numpy as NPY
  print(MUD.__file__)

  # MUD.f_double(1)

  face_vtx_32  = NPY.empty(10, dtype='int32', order='F')
  face_vtx_64  = NPY.empty(10, dtype='int64', order='F')
  face_vtx_idx = NPY.empty(3 , dtype='int32', order='F')

  face_vtx_u64  = NPY.empty(10, dtype='double', order='F')

  MUD.auto_dispatch(face_vtx_32, face_vtx_idx)
  MUD.auto_dispatch(face_vtx_64, face_vtx_idx)

  print(help(MUD.auto_dispatch))

  try:
    MUD.auto_dispatch(face_vtx_u64, face_vtx_idx)
  except TypeError:
    print("Bad type pass")
    pass
