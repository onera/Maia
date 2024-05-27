import pytest
import pytest_parallel
import os
import numpy as np
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT

import maia.utils.test_utils as TU

from maia.io import meshb_converter, \
                    file_to_dist_tree

def test_get_tree_info():

  dist_tree = maia.factory.generate_dist_block(11, 'TETRA_4', MPI.COMM_SELF)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
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
    vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
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

    assert int(lines[lines.index('Vertices\n')+1])       == 9474
    assert int(lines[lines.index('Triangles\n')+1])      == 3024
    assert int(lines[lines.index('Quadrilaterals\n')+1]) == 630
    assert int(lines[lines.index('Tetrahedra\n')+1])     == 21450
    assert int(lines[lines.index('Prisms\n')+1])         == 9320
    assert lines[-1]                                     == 'End\n'

    # ---- Check BCs for triangles
    st_triangles             = lines.index('Triangles\n')
    tri_bc_tag               = [int(l.split()[-1]) for l in lines[st_triangles+2:st_triangles+2+3024]]
    tri_u_tag, tri_bc_counts = np.unique(tri_bc_tag, return_counts=True)

    assert (tri_u_tag == [1, 2, 3, 4, 5, 6, 7, 8]).all()
    assert (sum(tri_bc_counts) == 3024)

    # ---- Check BCs for quadrilaterals
    st_quads                   = lines.index('Quadrilaterals\n')
    quad_bc_tag                = [int(l.split()[-1]) for l in lines[st_quads+2:st_quads+2+630]]
    quad_u_tag, quad_bc_counts = np.unique(quad_bc_tag, return_counts=True)

    assert(quad_u_tag == [2, 3, 4, 8]).all()
    assert(sum(quad_bc_counts) == 630)

    # ---- Check sol & metric
    with open(files['sol']) as f:
        lines = f.readlines()

    assert int(lines[lines.index('SolAtVertices\n')+1]) == 9474
    assert     lines[lines.index('SolAtVertices\n')+2]  == '1 1 \n'

    with open(files['fld']) as f:
        lines = f.readlines()

    assert int(lines[lines.index('SolAtVertices\n')+1]) == 9474
    assert     lines[lines.index('SolAtVertices\n')+2]  == '2 1 1 \n'


@pytest_parallel.mark.parallel(2)
def test_meshb_to_cgns(comm):
  # Prepare test : write files in serial
  tmp_dir = TU.create_collective_tmp_dir(comm)
  files = {'mesh': tmp_dir / 'mesh.mesh',
           'fld' : tmp_dir / 'field.sol'}

  if comm.Get_rank() == 0:
    dist_tree = maia.factory.generate_dist_block(11, 'TETRA_4', MPI.COMM_SELF)
    zone = PT.get_all_Zone_t(dist_tree)[0]

    vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
    n_vtx = vtx_distri[1] - vtx_distri[0]
    fields = {"Zeros": np.zeros(n_vtx), "Range": np.arange(n_vtx, dtype=float)}
    PT.new_FlowSolution('FlowSolution', loc='Vertex', fields=fields, parent=zone)

    meshb_converter.cgns_to_meshb(dist_tree, files, [], ['FlowSolution'], constraints=None)

  tree_info = {
               'bc_names': {
                   'EdgeCenter' : [],
                   'FaceCenter' : ['bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6'],
                   'CellCenter' : [],
                   },
               'field_names' : { 'FlowSolution' : ['Zeros', 'Range'] },
              }

  dist_tree = meshb_converter.meshb_to_cgns(files, tree_info, comm)

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert PT.Zone.n_vtx(zone) == 1331 and PT.Zone.n_cell(zone) == 5000

  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
  sol = PT.get_node_from_path(zone, 'FlowSolution/Range')[1]
  assert (sol == np.arange(vtx_distri[0], vtx_distri[1])).all()

  # TODO BCs are poorly distributed
  bc = PT.get_node_from_name(zone, 'bc3')
  assert PT.maia.getDistribution(bc, 'Index')[1][2] == 200
