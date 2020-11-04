from   mpi4py             import MPI
import Converter.Internal as     I
from . import cgns_registry     as     CGR
from . import cgns_keywords     as     CGK

def build_paths_by_label_bcdataset(paths_by_label, bc, bc_path):
  """
  Factorize BC+GC
  """
  for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
    bcds_path = bc_path+"/"+I.getName(bcds)
    CGR.add_path(paths_by_label, bcds_path, "BCDataSet_t")
    for bcd in I.getNodesFromType1(bcds, 'BCData_t'):
      bcd_path = bcds_path+"/"+I.getName(bcd)
      CGR.add_path(paths_by_label, bcd_path, "BCData_t")

def build_paths_by_label_zone(paths_by_label, zone, zone_path):
  """
  """
  for zone_bc in  I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+I.getName(zone_bc)
    CGR.add_path(paths_by_label, zone_bc_path, "ZoneBC_t")
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+I.getName(bc)
      CGR.add_path(paths_by_label, bc_path, "BC_t")
      build_paths_by_label_bcdataset(paths_by_label, bc, bc_path)

  for zone_gc in  I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+I.getName(zone_gc)
    CGR.add_path(paths_by_label, zone_gc_path, "ZoneGridConnectivity_t")

    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
      gc_path = zone_gc_path+"/"+I.getName(gc)
      CGR.add_path(paths_by_label, gc_path, "GridConnectivity_t")
      build_paths_by_label_bcdataset(paths_by_label, gc, gc_path)

    for gc1to1 in I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t'):
      gc1to1_path = zone_gc_path+"/"+I.getName(gc1to1)
      # > Here is a little hack to have gridConnectivity in same label
      CGR.add_path(paths_by_label, gc1to1_path, "GridConnectivity_t")
      build_paths_by_label_bcdataset(paths_by_label, gc1to1, gc1to1_path)

def setup_child_from_type(paths_by_label, parent, parent_path, cgns_type):
  """
  """
  for child in I.getNodesFromType1(parent, cgns_type):
    child_path = parent_path+"/"+I.getName(child)
    CGR.add_path(paths_by_label, child_path, "Family_t")

def build_paths_by_label(tree):
  """
  """
  paths_by_label = CGR.cgns_paths_by_label();

  bases_path = I.getPathsFromType1(tree, 'CGNSBase_t', pyCGNSLike=True)

  for base in I.getNodesFromType1(tree, 'CGNSBase_t'):
    base_path = "/"+I.getName(base)
    CGR.add_path(paths_by_label, base_path, u'CGNSBase_t')

    setup_child_from_type(paths_by_label, base, base_path, 'Family_t')
    setup_child_from_type(paths_by_label, base, base_path, 'FlowEquationSet_t')
    setup_child_from_type(paths_by_label, base, base_path, 'ViscosityModel_t')
    # for fam in I.getNodesFromType1(base, 'Family_t'):
    #   fam_path = "/"+I.getName(base)+"/"+I.getName(fam)
    #   CGR.add_path(paths_by_label, fam_path, "Family_t")

    for zone in I.getNodesFromType1(base, 'Zone_t'):
      zone_path = "/"+I.getName(base)+"/"+I.getName(zone)
      CGR.add_path(paths_by_label, zone_path, 'Zone_t')
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








