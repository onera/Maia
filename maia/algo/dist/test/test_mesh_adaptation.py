import pytest
import pytest_parallel
import shutil

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import mesh_adaptation as MA

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np

feflo_exists = shutil.which('feflo.a') is not None
mmg3d_exists = True
try:
  MA.detect_mmg(3)
except FileNotFoundError:
  mmg3d_exists = False

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
        Two2DTensorXY DataArray_t R8 [1., 1., 1.]:
        Two2DTensorYY DataArray_t R8 [1., 1., 1.]:
        Two2DTensorXX DataArray_t R8 [1., 1., 1.]:
  """
  tree = PT.yaml.to_cgns_tree(yz)

  # > Wrong because leads to unexistant field
  with pytest.raises(ValueError):
    MA.unpack_metric(tree, "FlowSol/toto")

  # > Wrong because leads to 3 fields
  with pytest.raises(ValueError):
    MA.unpack_metric(tree, "FlowSol/Wrong")

  # > Path to unique field
  metrics_names = [PT.get_name(n) for n in MA.unpack_metric(tree, "FlowSol/Mach")]
  assert metrics_names==["Mach"]

  # > Isotrop metric
  assert MA.unpack_metric(tree, None) == []

  # > Path to multiple fields
  metrics_names = [PT.get_name(n) for n in MA.unpack_metric(tree, "FlowSol/Tensor")]
  assert metrics_names==["TensorXX","TensorXY","TensorXZ",
                        "TensorYY","TensorYZ","TensorZZ"]

  # > Path to multiple fields (2D, so only XX,XY,YY)
  metrics_names = [PT.get_name(n) for n in MA.unpack_metric(tree, "FlowSol/Two2DTensor")]
  assert metrics_names==["Two2DTensorXX", "Two2DTensorXY","Two2DTensorYY"]

  # > Wrong because metric path is integer (not str and not list)
  with pytest.raises(ValueError):
    MA.unpack_metric(tree, 10)

  # > Wrong because leads to scalar and tensor fields (4 or 7 nodes)
  tree_extra = PT.deep_copy(tree)
  fs = PT.get_node_from_name(tree_extra, 'FlowSol')
  PT.new_DataArray('Tensor'     , [1., 1., 1.], parent=fs)
  PT.new_DataArray('Two2DTensor', [1., 1., 1.], parent=fs)

  with pytest.raises(ValueError):
    MA.unpack_metric(tree_extra, "FlowSol/Tensor")
  with pytest.raises(ValueError):
    MA.unpack_metric(tree_extra, "FlowSol/Two2DTensor")

  # > Paths to multiple fields (order matters)
  metric = ["FlowSol/TensorXX", "FlowSol/TensorZZ",
            "FlowSol/TensorXZ", "FlowSol/TensorXY",
            "FlowSol/TensorYZ", "FlowSol/TensorYY"]
  metrics_names = [PT.get_name(n) for n in MA.unpack_metric(tree, metric)]
  assert metrics_names==[ "TensorXX","TensorZZ","TensorXZ",
                          "TensorXY","TensorYZ","TensorYY" ]


  # Wrong : tensor with only two fields
  tree_extra = PT.yaml.to_cgns_tree("""
  Base CGNSBase_t [3, 3]:
    Zone Zone_t [[3,1,0]]:
      ZoneType ZoneType_t "Unstructured":
      FlowSol FlowSolution_t:
        TensorXY DataArray_t R8 [1., 1., 1.]:
        TensorYZ DataArray_t R8 [1., 1., 1.]:
  """)
  with pytest.raises(ValueError):
    MA.unpack_metric(tree_extra, "FlowSol/Tensor")



@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('multi_elt' , [False, True])
@pytest.mark.parametrize('custom_dir', [False, True])
def test_adapt_with_feflo(comm, multi_elt, custom_dir):
  import os
  import maia.utils.test_utils as TU

  if multi_elt:
    yaml_path = os.path.join(TU.mesh_dir, 'multi_element.yaml')
    dist_tree = maia.io.file_to_dist_tree(yaml_path, comm)
  else:
    dist_tree = maia.factory.generate_dist_block(5, 'TETRA_4', comm)

  base = PT.get_node_from_label(dist_tree, 'CGNSBase_t')
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  PT.set_name(zone, 'MyZone')

  # > To check meshb_reader after feflo since it doesn't preserve volumic BCs when multiple 3d elements
  if not multi_elt:
    zone_bc = PT.get_node_from_label(zone, 'ZoneBC_t')
    cell_distrib = MT.Zone.cell_distribution(zone)
    cell_pl = np.arange(cell_distrib[0], cell_distrib[1], dtype=pdm_gnum_dtype).reshape((1,-1), order='F')+1
    cell_bc = PT.new_BC("vol_bc", type="BCWall", loc="CellCenter", point_list=cell_pl, parent=zone_bc)
    MT.new_Distribution({"Index":cell_distrib}, parent=cell_bc)

    bc = PT.get_node_from_name(zone, 'Xmin')
    PT.set_value(bc, 'FamilySpecified')
    PT.new_child(bc, 'FamilyName', 'FamilyName_t', 'SomeFamily')
    PT.new_child(base, 'SomeFamily', 'Family_t')

  # > Create a metric field
  cx, cy, cz = PT.Zone.coordinates(zone)
  fields= {'metric' : (cx-0.5)**5+(cy-0.5)**5 - 1}
  PT.new_FlowSolution("FlowSolution", loc="Vertex", fields=fields, parent=zone)

  # > Adapt mesh according to scalar metric
  options = {"tmp_dir":"tmp_dir"} if custom_dir else {}
  adpt_dist_tree = MA.adapt_mesh_with_feflo(dist_tree,
                                            "FlowSolution/metric",
                                            comm,
                                            containers_name='ALL',
                                            feflo_opts="-c 100 -cmax 100 -p 4",
                                            **options)

  # Parsing of meshb is already tested elsewhere, here we check that feflo did not failed
  # and that metadata (eg. names, families) are well recovered
  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  assert PT.get_name(adpt_zone) == 'MyZone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(zone)

  is_cell_bc = PT.pred.is_bc_of_location('CellCenter')
  if multi_elt:
    assert len(PT.get_nodes_from_predicate(adpt_dist_tree, is_cell_bc))==0
  else:
    adpt_bc = PT.get_node_from_name(adpt_zone, 'Xmin')
    assert PT.get_value(adpt_bc) == 'FamilySpecified'
    assert PT.get_value(PT.get_child_from_name(adpt_bc, 'FamilyName')) == 'SomeFamily'
    assert PT.get_node_from_name_and_label(adpt_dist_tree, 'SomeFamily', 'Family_t') is not None

    cell_bc_nodes = PT.get_nodes_from_predicate(adpt_dist_tree, is_cell_bc)
    assert len(PT.get_nodes_from_predicate(adpt_dist_tree, is_cell_bc))==1
    assert PT.get_name(cell_bc_nodes[0])=='vol_bc'


@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
@pytest_parallel.mark.parallel(2)
def test_periodic_adapt_with_feflo(comm):

  # > Create simple mesh
  dist_tree = maia.factory.generate_dist_block(3, 'TETRA_4', comm)
  PT.rm_nodes_from_name(dist_tree, 'NODE*')

  # > Define metric
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  dn_vtx = MT.Zone.dn_vtx(dist_zone)
  fld_metric = np.ones(dn_vtx, dtype=float)
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
                                                        containers_name=['Metric'],
                                                        periodic=True,
                                                        feflo_opts=f"-c 10 -cmax 10 -p 4")

  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  for bc_name in ['Ymin','Ymax','Zmin','Zmax']:
    assert PT.get_node_from_name(adpt_zone, bc_name) is not None
  assert PT.get_name(adpt_zone) == 'zone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(dist_zone)
  adpt_gc = PT.get_node_from_name(adpt_zone, 'Xmin_0')
  assert PT.get_value(PT.get_child_from_name(adpt_gc, 'GridConnectivityDonorName')) == 'Xmax_0'

@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
@pytest_parallel.mark.parallel(4)
def test_periodic_adapt_with_feflo_axisym(comm):

  # > Read axisym mesh
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', comm)

  # > Define metric
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  h_min = 0.01 ; h_max = 1.
  r = np.sqrt(cy**2+cz**2)
  fld_metric = h_min + (h_max - h_min) * 0.5 * (10. + np.cos(1.*(r)))
  fld_metric[np.logical_and(0.3<r,r<0.4)] *= 10.
  PT.new_FlowSolution('Metric', loc='Vertex', fields={'metric':fld_metric}, parent=dist_zone)

  # > Periodic adaptation
  adpt_dist_tree = maia.algo.dist.adapt_mesh_with_feflo(dist_tree,
                                                        'Metric/metric',
                                                        comm,
                                                        containers_name=['Metric'],
                                                        periodic=True,
                                                        feflo_opts=f"-c 10 -cmax 10 -p 4")

  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  for bc_name in ['in','out','top']+[f'ridge.{i}' for i in range(9)]:
    assert PT.get_node_from_name(adpt_zone, bc_name) is not None
  assert PT.get_name(adpt_zone) == 'zone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(dist_zone)
  adpt_gc = PT.get_node_from_name(adpt_zone, 'per0_0')
  assert PT.get_value(PT.get_child_from_name(adpt_gc, 'GridConnectivityDonorName')) == 'per1_0'


@pytest.mark.skipif(not mmg3d_exists, reason="Require mmg3d")
@pytest_parallel.mark.parallel(2)
def test_adapt_with_mmg(comm):
  dist_tree = maia.factory.generate_dist_block(5, 'TETRA_4', comm)

  base = PT.get_node_from_label(dist_tree, 'CGNSBase_t')
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  PT.set_name(zone, 'MyZone')

  # > To check meshb_reader after mmg3d since it doesn't preserve volumic BCs when multiple 3d elements
  zone_bc = PT.get_node_from_label(zone, 'ZoneBC_t')
  cell_distrib = MT.Zone.cell_distribution(zone)
  cell_pl = np.arange(cell_distrib[0], cell_distrib[1], dtype=pdm_gnum_dtype).reshape((1,-1), order='F')+1
  cell_bc = PT.new_BC("vol_bc", type="BCWall", loc="CellCenter", point_list=cell_pl, parent=zone_bc)
  MT.new_Distribution({"Index":cell_distrib}, parent=cell_bc)

  bc = PT.get_node_from_name(zone, 'Xmin')
  PT.set_value(bc, 'FamilySpecified')
  PT.new_child(bc, 'FamilyName', 'FamilyName_t', 'SomeFamily')
  PT.new_child(base, 'SomeFamily', 'Family_t')

  # > Create a metric field
  cx, _, _ = PT.Zone.coordinates(zone)
  metric = np.ones_like(cx)
  metric[cx <  0.5] = 0.1
  metric[cx >= 0.5] = 0.2
  PT.new_FlowSolution("FlowSolution", loc="Vertex", fields={'metric' : metric}, parent=zone)

  # > Adapt mesh according to scalar metric
  adpt_dist_tree = MA.adapt_mesh_with_mmg(dist_tree,
                                          "FlowSolution/metric",
                                          False,
                                          comm,
                                          mmg_opts="-hgrad 1.2")

  # Parsing of meshb is already tested elsewhere, here we check that mmg did not failed
  # and that metadata (eg. names, families) are well recovered
  adpt_zone = PT.get_all_Zone_t(adpt_dist_tree)[0]
  assert PT.get_name(adpt_zone) == 'MyZone'
  assert PT.Zone.n_vtx(adpt_zone) != PT.Zone.n_vtx(zone)

  adpt_bc = PT.get_node_from_name(adpt_zone, 'Xmin')
  assert PT.get_value(adpt_bc) == 'FamilySpecified'
  assert PT.get_value(PT.get_child_from_name(adpt_bc, 'FamilyName')) == 'SomeFamily'
  assert PT.get_node_from_name_and_label(adpt_dist_tree, 'SomeFamily', 'Family_t') is not None

  is_cell_bc = PT.pred.is_bc_of_location('CellCenter')
  cell_bc_nodes = PT.get_nodes_from_predicate(adpt_dist_tree, is_cell_bc)
  assert len(PT.get_nodes_from_predicate(adpt_dist_tree, is_cell_bc)) == 1
  assert PT.get_name(cell_bc_nodes[0]) == 'vol_bc'