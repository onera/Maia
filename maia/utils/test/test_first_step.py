
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

  print(zone_u1.big_vector[400])
  print(zone_s1.big_vector[400])
  MUF.add_zone_to_base(base, zone_u1);
  MUF.add_zone_to_base(base, zone_s1);

  # > En faite ce qui est pas mal c'est que pybind retrouve le bon type caché par le variant
  # > Ok ca marche
  zone_t = MUF.get_zone_from_gid(base, 1)
  print(zone_t)
  print(zone_t.ngon) # type == zone_unstructured

  zone_ts = MUF.get_zone_from_gid(base, 2)
  print(zone_ts)
  print(zone_ts.zone_opp_name) # type == zone_structured

  # Si la zone est pas trouvé --> None
  # Si pas le type de retour n'est pas wrappé --> Cannot convert C++ object to python -> Logique, il sait pas faire le translate

  # > C'est bizarre ca, car la fonction accepte indiremment un pointer ou adress ??
  # MUF.add_zone_to_base(base, zone_t);

  zone_u1.global_id = 1000 # Well this one works but not for good reason

  print("python side before del : ")
  print(zone_u1.global_id)
  print(zone_s1.global_id)
  # print(zone_u1.big_vector[400]) # Fail
  # print(zone_s1.big_vector[400]) # Fail
  print(zone_ts.big_vector[400])   # > Ok because we retake the zone by accessor
  print(base)
  print("python side before del : ")

  del(zone_u1)
  del(zone_s1)

  print(base)

  del(base)

  # > Mega core dump, donc la mémoire degage bien
  # print(zone_ts.global_id)
  # print(zone_ts.big_vector[400])   # > Ok because we retake the zone by accessor


