import pytest

# --------------------------------------------------------------------------
# @pytest.mark.parametrize("memorder", 1, 2, 3, ids=lambda fixture_value:fixture_value.name)
@pytest.mark.parametrize("memorder", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float", "double", "int"])
def test_memory_key(dtype, memorder):
  print("dtype::"   ,dtype)
  print("memorder::",memorder)
