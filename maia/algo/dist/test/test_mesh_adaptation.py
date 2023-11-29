import pytest
import pytest_parallel
import shutil

import maia
import maia.pytree as PT
from maia.pytree.yaml          import parse_yaml_cgns

from maia.algo.dist import mesh_adaptation as MA

import numpy as np

feflo_exists = shutil.which('feflo.a') is not None

def test_unpack_metric():
  yz = """
  Base CGNSBase_t [3, 3]:
    Zone Zone_t [[3,1,0]]:
      ZoneType ZoneType_t "Unstructured":
      FlowSol FlowSolution_t:
        Mach     DataArray_t R8 [1., 1., 1.]:
        TensorXX DataArray_t R8 [1., 1., 1.]:
        TensorZZ DataArray_t R8 [1., 1., 1.]:
        TensorXY DataArray_t R8 [1., 1., 1.]:
        TensorYY DataArray_t R8 [1., 1., 1.]:
        TensorXZ DataArray_t R8 [1., 1., 1.]:
        TensorYZ DataArray_t R8 [1., 1., 1.]:
        WrongA   DataArray_t R8 [1., 1., 1.]:
        WrongB   DataArray_t R8 [1., 1., 1.]:
        WrongC   DataArray_t R8 [1., 1., 1.]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yz)

  # > Wrong because leads to unexistant field
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(MA.unpack_metric(tree, "FlowSol/toto"))

  # > Wrong because leads to 3 fields
  with pytest.raises(ValueError):
    metrics_names = PT.get_names(MA.unpack_metric(tree, "FlowSol/Wrong"))

  # > Path to unique field
  metrics_names = PT.get_names(MA.unpack_metric(tree, "FlowSol/Mach"))
  assert metrics_names==["Mach"]
  
  # > Isotrop metric
  assert MA.unpack_metric(tree, None) == []

  # > Path to multiple fields
  metrics_names = PT.get_names(MA.unpack_metric(tree, "FlowSol/Tensor"))
  assert metrics_names==[ "TensorXX","TensorXY","TensorXZ",
                          "TensorYY","TensorYZ","TensorZZ" ]

  # > Paths to multiple fields (order matters)
  metric = ["FlowSol/TensorXX", "FlowSol/TensorZZ",
            "FlowSol/TensorXZ", "FlowSol/TensorXY",
            "FlowSol/TensorYZ", "FlowSol/TensorYY"]
  metrics_names = PT.get_names(MA.unpack_metric(tree, metric))
  assert metrics_names==[ "TensorXX","TensorZZ","TensorXZ",
                          "TensorXY","TensorYZ","TensorYY" ]

@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
@pytest_parallel.mark.parallel(2)
def test_adapt_with_feflo(comm):

  dist_tree = maia.factory.generate_dist_block(5, 'TETRA_4', comm)
  base = PT.get_node_from_label(dist_tree, 'CGNSBase_t')
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  PT.set_name(zone, 'MyZone')
  bc = PT.get_node_from_name(zone, 'Xmin')
  PT.set_value(bc, 'FamilySpecified')
  PT.new_child(bc, 'FamilyName', 'FamilyName_t', 'SomeFamily')

  PT.new_child(base, 'SomeFamily', 'Family_t')

  # > Create a metric field
  cx, cy, cz = PT.Zone.coordinates(zone)
  fields= {'metric' : (cx-0.5)**5+(cy-0.5)**5 - 1}
  PT.new_FlowSolution("FlowSolution", loc="Vertex", fields=fields, parent=zone)

  # > Adapt mesh according to scalar metric
  adpt_dist_tree = MA.adapt_mesh_with_feflo(dist_tree,
                                            "FlowSolution/metric",
                                            comm,
                                            container_names=["FlowSolution"],
                                            feflo_opts="-c 100 -cmax 100 -p 4")

  # Parsing of meshb is already tested elsewhere, here we check that feflo did not failed 
  # and that metadata (eg. names, families) are well recovered
  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  assert PT.get_name(adpt_zone) == 'MyZone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(zone)
  adpt_bc = PT.get_node_from_name(adpt_zone, 'Xmin')
  assert PT.get_value(adpt_bc) == 'FamilySpecified'
  assert PT.get_value(PT.get_child_from_name(adpt_bc, 'FamilyName')) == 'SomeFamily'
  assert PT.get_node_from_name_and_label(adpt_dist_tree, 'SomeFamily', 'Family_t') is not None

@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
@pytest_parallel.mark.parallel(2)
def test_periodic_adapt_with_feflo(comm):

  # > Create simple mesh
  dist_tree = maia.factory.dcube_generator.dcube_nodal_generate(3, 1., np.array([0.,0.,0.], dtype=np.float64), 'TETRA_4', comm, get_ridges=False)
  PT.rm_nodes_from_name(dist_tree, 'NODE*')

  # > Define metric
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  n_vtx = PT.Zone.n_vtx(dist_zone)
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  fld_metric = np.ones(n_vtx, dtype=np.float64)
  PT.new_FlowSolution('Metric', loc='Vertex', fields={'metric':fld_metric}, parent=dist_zone)

  # > Build periodicities
  zone_bc_n = PT.get_node_from_label(dist_tree, 'ZoneBC_t')
  for bc_name in ['Xmin', 'Xmax']:
    bc_n  = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.new_node('FamilyName', label='FamilyName_t', value=bc_name.upper(), parent=bc_n)
  periodic = {'translation' : np.array([1.0, 0, 0], np.float32)}
  maia.algo.dist.connect_1to1_families(dist_tree, ('XMIN', 'XMAX'), comm, periodic=periodic, location='Vertex')
  assert len(PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t'))!=0

  # > Periodic adaptation
  adpt_dist_tree = maia.algo.dist.adapt_mesh_with_feflo(dist_tree,
                                                        'Metric/metric',
                                                        comm,
                                                        container_names=['Metric'],
                                                        periodic=True,
                                                        feflo_opts=f"-c 10 -cmax 10 -p 4")
  
  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  for bc_name in ['Ymin','Ymax','Zmin','Zmax']:
    assert PT.get_node_from_name(adpt_zone, bc_name) is not None
  assert PT.get_name(adpt_zone) == 'zone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(dist_zone)
  adpt_gc = PT.get_node_from_name(adpt_zone, 'Xmin_0')
  assert PT.get_value(PT.get_child_from_name(adpt_gc, 'GridConnectivityDonorName')) == 'Xmax_0'