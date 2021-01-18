import pytest
import itertools as ITT

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

  # MUF.add_zone_to_base(base, zone_u1);
  # MUF.add_zone_to_base(base, zone_s1);

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
  assert_mpi(comm, 0, comm.rank == 0)
  assert_mpi(comm, 1, comm.rank == 1)

# --------------------------------------------------------------------------
@pytest.mark.mpi
def test_size():
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert comm.size > 0

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

