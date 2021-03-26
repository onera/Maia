import pytest
import itertools as ITT

# --------------------------------------------------------------------------
def test_first_step():
  """
  """
  from maia.__unused.test_pybind import first_step as MUF

  base = MUF.cgns_base("cgns_base", 1)

  zone_u1 = MUF.zone_unstructured("zone_name", 1)
  assert(zone_u1.global_id == 1          );
  assert(zone_u1.name      == "zone_name");

  zone_s1 = MUF.zone_structured("cartesian", 2)
  assert(zone_s1.global_id == 2          );
  assert(zone_s1.name      == "cartesian");

  # print(zone_u1.big_vector[400])
  # print(zone_s1.big_vector[400])
  MUF.move_zone_to_base(base, zone_u1);
  MUF.move_zone_to_base(base, zone_s1);

  # > En faite ce qui est pas mal c'est que pybind retrouve le bon type caché par le variant
  # > Ok ca marche
  zone_t = MUF.get_zone_from_gid(base, 1)
  # print(zone_t)
  # print(zone_t.ngon) # type == zone_unstructured

  zone_ts = MUF.get_zone_from_gid(base, 2)
  # print(zone_ts)
  # print(zone_ts.zone_opp_name) # type == zone_structured

  # Si la zone est pas trouvé --> None
  # Si pas le type de retour n'est pas wrappé --> Cannot convert C++ object to python -> Logique, il sait pas faire le translate

  # > C'est bizarre ca, car la fonction accepte indiremment un pointer ou adress ??
  # MUF.move_zone_to_base(base, zone_t);

  zone_u1.global_id = 1000 # Well this one works but not for good reason

  # print("python side before del : ")
  # print(zone_u1.global_id)
  # print(zone_s1.global_id)
  # print(zone_u1.big_vector[400]) # Fail
  # print(zone_s1.big_vector[400]) # Fail
  # print(zone_ts.big_vector[400])   # > Ok because we retake the zone by accessor
  # print(base)
  # print("python side before del : ")

  del(zone_u1)
  del(zone_s1)

  # print(base)

  del(base)

  # > Mega core dump, donc la mémoire degage bien
  # print(zone_ts.global_id)
  # print(zone_ts.big_vector[400])   # > Ok because we retake the zone by accessor

def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4
    assert inc(4) == 5
    assert inc(5) == 6
    # assert inc(6) == 8

# --------------------------------------------------------------------------
def assert_mpi(comm, rank, cond ):
  if(comm.rank == rank):
    print(cond)
    assert(cond == True)
  else:
    pass

# --------------------------------------------------------------------------
def test_first_step():
  """
  """
  # print("test_first_step")
  from maia.__unused.test_pybind import first_step as MUF

  base = MUF.cgns_base("cgns_base", 1)

  zone_u1 = MUF.zone_unstructured("zone_name", 1)
  assert(zone_u1.global_id == 1          );
  assert(zone_u1.name      == "zone_name");

  zone_s1 = MUF.zone_structured("cartesian", 2)
  assert(zone_s1.global_id == 2          );
  assert(zone_s1.name      == "cartesian");

  # MUF.move_zone_to_base(base, zone_u1);
  # MUF.move_zone_to_base(base, zone_s1);

  print(zone_u1.global_id)
  print(zone_s1.global_id)
  print(base)

# --------------------------------------------------------------------------
def test_mpi():
  """
  """
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  print("hello", comm.size)
  assert comm.size > 0
  # assert(0 == 1)
  assert_mpi(comm, 0, comm.rank == 0)
  assert_mpi(comm, 1, comm.rank == 1)

# --------------------------------------------------------------------------
# @pytest.mark.mpi
# def test_size():
#   from mpi4py import MPI
#   comm = MPI.COMM_WORLD
#   assert comm.size > 0

# --------------------------------------------------------------------------
#@pytest.mark.mpi(min_size=2)
#def test_size():
#  from mpi4py import MPI
#  comm = MPI.COMM_WORLD
#  print("hello", comm.size, comm.rank)
#  assert comm.size >= 2


# --------------------------------------------------------------------------
# @pytest.mark.parametrize("memorder", 1, 2, 3, ids=lambda fixture_value:fixture_value.name)
@pytest.mark.parametrize("memorder", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float", "double", "int"])
def test_memory_key(dtype, memorder):
  print("dtype::"   ,dtype)
  print("memorder::",memorder)



