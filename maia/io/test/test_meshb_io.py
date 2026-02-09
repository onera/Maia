import pytest
import pytest_parallel
import os
import numpy as np
import mpi4py.MPI as MPI

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

import maia.utils.test_utils as TU

from maia.io import meshb_converter, \
                    file_to_dist_tree

def test_get_tree_info():

  dist_tree = maia.factory.generate_dist_block(11, 'TETRA_4', MPI.COMM_SELF)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  vtx_distri = MT.Zone.vtx_distribution(zone)
  n_vtx = vtx_distri[1] - vtx_distri[0]
  fields = {"Zeros": np.zeros(n_vtx), "Range": np.arange(n_vtx, dtype=float)}
  PT.new_FlowSolution('FlowSolution', loc='Vertex', fields=fields, parent=zone)

  # Families are required
  for ibc, bc in enumerate(PT.get_nodes_from_label(zone, 'BC_t')):
    PT.new_child(bc, 'FamilyName', 'FamilyName_t', f'fam{ibc+1}')

  tree_info = meshb_converter.get_tree_info(dist_tree, ['FlowSolution'])

  assert len(tree_info) == 2
  assert tree_info['field_names'] == {'FlowSolution' : ['Zeros', 'Range']}
  assert tree_info['bc_names'] == {
          'EdgeCenter': [],
          'FaceCenter': ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax'],
          'CellCenter': [],
          }


def test_cgns_to_meshb(tmp_path):
    # ---- Loading yaml/cgns mesh file
    yaml_path = os.path.join(TU.mesh_dir, 'multi_element.yaml')
    dist_tree = file_to_dist_tree(yaml_path, MPI.COMM_SELF)

    # ---- Setting up flow solution
    zone       = PT.get_all_Zone_t(dist_tree)[0]
    vtx_distri = MT.Zone.vtx_distribution(zone)
    n_vtx      = vtx_distri[1] - vtx_distri[0]

    fields     = {
        'Zeros': np.zeros(n_vtx),
        'Range': np.arange(n_vtx, dtype=float)
    }

    PT.new_FlowSolution('FlowSolution', loc='Vertex', fields=fields, parent=zone)
    PT.new_FlowSolution('Metric', loc='Vertex', fields={'Ones': np.ones(n_vtx)}, parent=zone)

    # ---- Write mesh & sol to mesh format
    files = {
        'mesh': tmp_path / 'multi_element.mesh',
        'sol' : tmp_path / 'metric.sol',
        'fld' : tmp_path / 'field.sol'
    }

    meshb_converter.cgns_to_meshb(
        dist_tree, files,
        [PT.get_node_from_name(zone, 'Ones')],
        ['FlowSolution'],
        constraints=None
    )

    # ---- Check mesh
    with open(files['mesh']) as f:
        lines = f.readlines()

    assert int(lines[lines.index('Vertices\n')+1])       == 77
    assert int(lines[lines.index('Edges\n')+1])          == 56
    assert int(lines[lines.index('Triangles\n')+1])      == 88
    assert int(lines[lines.index('Quadrilaterals\n')+1]) == 16
    assert int(lines[lines.index('Tetrahedra\n')+1])     == 144
    assert int(lines[lines.index('Prisms\n')+1])         == 24
    assert lines[-1]                                     == 'End\n'

    # ---- Check BCs for triangles
    st_triangles             = lines.index('Triangles\n')
    tri_bc_tag               = [int(l.split()[-1]) for l in lines[st_triangles+2:st_triangles+2+88]]
    tri_u_tag, tri_bc_counts = np.unique(tri_bc_tag, return_counts=True)

    assert (tri_u_tag == [1, 2, 3, 4, 5, 6]).all()
    assert (sum(tri_bc_counts) == 88)

    # ---- Check BCs for quadrilaterals
    st_quads                   = lines.index('Quadrilaterals\n')
    quad_bc_tag                = [int(l.split()[-1]) for l in lines[st_quads+2:st_quads+2+16]]
    quad_u_tag, quad_bc_counts = np.unique(quad_bc_tag, return_counts=True)

    assert(quad_u_tag == [1, 2, 5, 6]).all()
    assert(sum(quad_bc_counts) == 16)

    # ---- Check sol & metric
    with open(files['sol']) as f:
        lines = f.readlines()

    assert int(lines[lines.index('SolAtVertices\n')+1]) == 77
    assert     lines[lines.index('SolAtVertices\n')+2]  == '1 1 \n'

    with open(files['fld']) as f:
        lines = f.readlines()

    assert int(lines[lines.index('SolAtVertices\n')+1]) == 77
    assert     lines[lines.index('SolAtVertices\n')+2]  == '2 1 1 \n'


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('multi_elt', [False, True])
def test_meshb_to_cgns(multi_elt, comm):
  # Prepare test : write files in serial
  tmp_dir = TU.create_collective_tmp_dir(comm)
  files = {'mesh': tmp_dir / 'mesh.mesh',
           'fld' : tmp_dir / 'field.sol'}

  if multi_elt:
    yaml_path = os.path.join(TU.mesh_dir, 'multi_element.yaml')
    dist_tree = file_to_dist_tree(yaml_path, comm)
    bc_edge_groups = ['bce'+str(idx+1) for idx in range(28)]
    bc_face_groups = ['bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6']
    bc_cell_groups = ['bcv1', 'bcv2']
  else:
    dist_tree = maia.factory.generate_dist_block(11, 'TETRA_4', comm)
    bc_edge_groups = []
    bc_face_groups = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
    bc_cell_groups = []

  zone = PT.get_all_Zone_t(dist_tree)[0]
  dn_vtx = MT.Zone.dn_vtx(zone)
  fields = {"Zeros": np.zeros(dn_vtx), "Range": np.arange(dn_vtx, dtype=float)}
  PT.new_FlowSolution('FlowSolution', loc='Vertex', fields=fields, parent=zone)

  # For comparaison
  dist_tree_bck = PT.deep_copy(dist_tree)

  # For meshb converter
  maia.algo.dist.redistribute_tree(dist_tree, 'gather.0', comm)
  if comm.Get_rank() == 0:
    meshb_converter.cgns_to_meshb(dist_tree, files, [], ['FlowSolution'], constraints=None)

  tree_info = {
               'bc_names': {
                   'EdgeCenter' : bc_edge_groups,
                   'FaceCenter' : bc_face_groups,
                   'CellCenter' : bc_cell_groups,
                   },
               'field_names' : { 'FlowSolution' : ['Zeros', 'Range'] },
              }

  meshb_dist_tree = meshb_converter.meshb_to_cgns(files, tree_info, comm)
  PT.rm_nodes_from_name(dist_tree_bck, "maia_topo") # Added by meshb -> cgns
  PT.rm_nodes_from_name(meshb_dist_tree, "maia_topo") # Added by meshb -> cgns

  # Compare on same distribution
  maia.algo.dist.redistribute_tree(dist_tree_bck, 'uniform', comm)
  maia.algo.dist.redistribute_tree(meshb_dist_tree, 'uniform', comm)

  if multi_elt:
    zone_n = PT.get_node_from_label(meshb_dist_tree, 'Zone_t')
    assert PT.Zone.n_vtx (zone_n)==77
    assert PT.Zone.n_cell(zone_n)==168
    n_elts = {'TETRA_4.0':144,'PENTA_6.1':24,'TRI_3.0':88,'QUAD_4.1':16,'BAR_2.0':56}
    for elt_name, n_elt in n_elts.items():
      elt_n = PT.get_node_from_name_and_label(zone_n, elt_name, 'Elements_t')
      assert MT.Element.distribution(elt_n)[2]==n_elt
    assert PT.get_node_from_name_and_label(meshb_dist_tree, 'bcv1', 'BC_t') is not None
    assert PT.get_node_from_name_and_label(meshb_dist_tree, 'bcv2', 'BC_t') is not None

  else:
      assert PT.is_same_tree(dist_tree_bck, meshb_dist_tree, abs_tol=1E-12)
  
  TU.rm_collective_dir(tmp_dir, comm)


