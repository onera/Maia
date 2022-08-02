from mpi4py import MPI

import Converter.Internal as I
import maia.pytree        as PT

from   cmaia.part_algo import cgns_registry as CGR
import maia.pytree.cgns_keywords as CGK
CGL = CGK.Label

def build_paths_by_label_bcdataset(paths_by_label, bc, bc_path):
  """
  Factorize BC+GC
  """
  for bcds in PT.get_children_from_label(bc, 'BCDataSet_t'):
    bcds_path = bc_path+"/"+I.getName(bcds)
    CGR.add_path(paths_by_label, bcds_path, "BCDataSet_t")
    for bcd in PT.get_children_from_label(bcds, 'BCData_t'):
      bcd_path = bcds_path+"/"+I.getName(bcd)
      CGR.add_path(paths_by_label, bcd_path, "BCData_t")

def build_paths_by_label_zone(paths_by_label, zone, zone_path):
  """
  """
  zone_bc = PT.request_node_from_label(zone, CGL.ZoneBC_t)
  if zone_bc is not None:
    zone_bc_path = f"{zone_path}/{I.getName(zone_bc)}"
    CGR.add_path(paths_by_label, zone_bc_path, CGL.ZoneBC_t.name)
    for bc in PT.get_children_from_label(zone_bc, 'BC_t'):
      bc_path = f"{zone_bc_path}/{PT.get_name(bc)}"
      CGR.add_path(paths_by_label, bc_path, "BC_t")
      build_paths_by_label_bcdataset(paths_by_label, bc, bc_path)

  for zone_gc in PT.iter_children_from_label(zone, CGL.ZoneGridConnectivity_t):
    zone_gc_path = f"{zone_path}/{PT.get_name(zone_gc)}"
    CGR.add_path(paths_by_label, zone_gc_path, CGL.ZoneGridConnectivity_t.name)

    for gc in PT.iter_children_from_label(zone_gc, CGL.GridConnectivity_t):
      gc_path = f"{zone_gc_path}/{PT.get_name(gc)}"
      CGR.add_path(paths_by_label, gc_path, CGL.GridConnectivity_t.name)
      build_paths_by_label_bcdataset(paths_by_label, gc, gc_path)

    for gc1to1 in PT.iter_children_from_label(zone_gc, CGL.GridConnectivity1to1_t):
      gc1to1_path = f"{zone_gc_path}/{PT.get_name(gc1to1)}"
      # > Here is a little hack to have gridConnectivity in same label
      CGR.add_path(paths_by_label, gc1to1_path, CGL.GridConnectivity_t.name)
      build_paths_by_label_bcdataset(paths_by_label, gc1to1, gc1to1_path)

def build_paths_by_label_family(paths_by_label, parent, parent_path):
  """
  """
  print(f"parent = {PT.get_name(parent), PT.get_label(parent)}, parent_path = {parent_path}")
  for family in PT.iter_children_from_label(parent, CGL.Family_t):
    if family != parent:
      family_path = f"{parent_path}/{PT.get_name(family)}"
      CGR.add_path(paths_by_label, family_path, CGL.Family_t.name)

      family_bc = PT.request_node_from_label(family, CGL.FamilyBC_t, depth=1)
      if family_bc is not None:
        family_bc_path = f"{family_path}/{I.getName(family_bc)}"
        CGR.add_path(paths_by_label, family_bc_path, CGL.FamilyBC_t.name)

        for family_bcdataset in PT.iter_children_from_label(family_bc, CGL.FamilyBCDataSet_t):
          family_bcdataset_path = f"{family_bc_path}/{PT.get_name(family_bcdataset)}"
          CGR.add_path(paths_by_label, family_bcdataset_path, CGL.FamilyBCDataSet_t.name)
          build_paths_by_label_bcdataset(paths_by_label, family_bcdataset, family_bcdataset_path)

      # Hierarchic family
      build_paths_by_label_family(paths_by_label, family, family_path)

def setup_child_from_type(paths_by_label, parent, parent_path, cgns_type):
  """
  """
  for child in PT.iter_children_from_label(parent, cgns_type):
    child_path = parent_path+"/"+I.getName(child)
    CGR.add_path(paths_by_label, child_path, cgns_type)

def build_paths_by_label(tree):
  """
  """
  paths_by_label = CGR.cgns_paths_by_label();

  for base in PT.get_all_CGNSBase_t(tree):
    base_path = "/"+I.getName(base)
    CGR.add_path(paths_by_label, base_path, u'CGNSBase_t')

    setup_child_from_type(paths_by_label, base, base_path, 'FlowEquationSet_t')
    setup_child_from_type(paths_by_label, base, base_path, 'ViscosityModel_t')

    build_paths_by_label_family(paths_by_label, base, base_path)

    for zone in PT.iter_nodes_from_label(base, CGL.Zone_t):
      zone_path = f"{base_path}/{PT.get_name(zone)}"
      CGR.add_path(paths_by_label, zone_path, CGL.Zone_t.name)
      build_paths_by_label_zone(paths_by_label, zone, zone_path)

  return paths_by_label

def make_cgns_registry(tree, comm):
  """
  Generate for each nodes a global identifier
  """
  paths_by_label = build_paths_by_label(tree)
  cgr = CGR.cgns_registry(paths_by_label, comm)
  return cgr


def add_cgns_registry_information(tree, comm):
  """
  """
  cgr = make_cgns_registry(tree, comm)

  for itype in range(CGK.nb_cgns_labels):
    paths      = cgr.paths(itype)
    global_ids = cgr.global_ids(itype)
    for i in range(len(paths)):
      # print(paths[i], global_ids[i])
      node    = I.getNodeFromPath(tree, paths[i])
      cgns_registry_n = I.getNodeFromNameAndType(node, ":CGNS#Registry", 'UserDefined_t')
      if cgns_registry_n:
        I._rmNode(node, cgns_registry_n)
      else:
        I.createNode(name=":CGNS#Registry", value=global_ids[i], ntype='UserDefined_t', parent=node)

  return cgr
