
# --------------------------------------------------------------------------
def test_first_step():
  """
  """
  from maia.utils import boundary_algorihms as MUB
  from maia.utils import first_step as MUF

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
