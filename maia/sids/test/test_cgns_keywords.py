import maia.sids.cgns_keywords as CGK

# print(f"dir(CGK) = {dir(CGK)}")
# print(f"CGK.Label) = {CGK.Label}")
# print(f"CGK.Label.Zone_t) = {CGK.Label.Zone_t}")
# print(f"CGK.Name) = {CGK.Name}")
# print(f"CGK.Name.GridCoordinates) = {CGK.Name.GridCoordinates}")
# print(f"type(CGK.Name.GridCoordinates) = {type(CGK.Name.GridCoordinates)}")
# print(f"CGK.GasModel) = {CGK.GasModel}")
# print(f"type(CGK.GasModel) = {type(CGK.GasModel)}")

def test_cgns_label():
  # print(f"dir(CGK.Label) = {dir(CGK.Label)}")
  # for label in CGK.Label.__members__:
  #   print(f"label = {label}")
  assert(isinstance(CGK.Label.CGNSTree_t, CGK.Label))

  assert(CGK.Label.CGNSTree_t.name  == "CGNSTree_t")
  assert(CGK.Label.CGNSBase_t.name  == "CGNSBase_t")
  assert(CGK.Label.Zone_t.name      == "Zone_t")
  assert(CGK.Label.ZoneBC_t.name    == "ZoneBC_t")
  assert(CGK.Label.BC_t.name        == "BC_t")
  assert(CGK.Label.BCDataSet_t.name == "BCDataSet_t")

  assert(CGK.Label.CGNSTree_t.value  == 0)
  assert(CGK.Label.CGNSBase_t.value  == 1)
  assert(CGK.Label.Zone_t.value      == 2)

  assert(CGK.Label.Zone_t.name in CGK.Label.__members__)

  assert(CGK.Label.__members__['CGNSBase_t'] == CGK.Label.CGNSBase_t)

def test_cgns_value():
  assert(CGK.GasModel.Ideal.name       == "Ideal")
  assert(CGK.GasModel.VanderWaals.name == "VanderWaals")

  assert(CGK.GasModel.Ideal.value       == 2)
  assert(CGK.GasModel.VanderWaals.value == 3)

def test_cgns_name():
  assert(CGK.Name.GridCoordinates == "GridCoordinates")
  assert(CGK.Name.GasModel        == "GasModel")
