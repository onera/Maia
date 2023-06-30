import pytest
import pytest_parallel
import numpy as np
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT

import maia.utils.test_utils as TU

from maia.io import meshb_converter

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
          'EdgeCenter' : [],
          'FaceCenter' :['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
          }

def test_cgns_to_meshb(tmp_path):

  # CGNS to mesh b is serial
  dist_tree = maia.factory.generate_dist_block(11, 'TETRA_4', MPI.COMM_SELF)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
  n_vtx = vtx_distri[1] - vtx_distri[0]
  fields = {"Zeros": np.zeros(n_vtx), "Range": np.arange(n_vtx, dtype=float)}
  PT.new_FlowSolution('FlowSolution', loc='Vertex', fields=fields, parent=zone)
  PT.new_FlowSolution('Metric', loc='Vertex', fields={"Ones": np.ones(n_vtx)}, parent=zone)

  files = {'mesh': tmp_path / 'mesh.mesh',
           'sol' : tmp_path / 'metric.sol',
           'fld' : tmp_path / 'field.sol'}

  meshb_converter.cgns_to_meshb(dist_tree, files, [PT.get_node_from_name(zone, 'Ones')], ['FlowSolution'])

  # Check .mesh
  with open(files['mesh']) as f:
    lines = f.readlines()

  assert int(lines[lines.index('Vertices\n')+1]) == 1331
  assert int(lines[lines.index('Edges\n')+1]) == 0
  assert int(lines[lines.index('Triangles\n')+1]) == 1200
  assert int(lines[lines.index('Tetrahedra\n')+1]) == 5000
  assert lines[-1] == "End\n"

  st_triangles = lines.index('Triangles\n')
  bc_tag = [int(l.split()[-1]) for l in lines[st_triangles+2:st_triangles+2+1200]]
  u_tag, counts = np.unique(bc_tag, return_counts=True)
  assert (u_tag == [1,2,3,4,5,6]).all()
  assert (counts == 200).all()

  # Check .sol
  with open(files['fld']) as f:
    lines = f.readlines()
  assert int(lines[lines.index('SolAtVertices\n')+1]) == 1331
  assert     lines[lines.index('SolAtVertices\n')+2]  == "2 1 1 \n"

  with open(files['sol']) as f:
    lines = f.readlines()
  assert int(lines[lines.index('SolAtVertices\n')+1]) == 1331
  assert     lines[lines.index('SolAtVertices\n')+2]  == "1 1 \n"


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
    
    meshb_converter.cgns_to_meshb(dist_tree, files, [], ['FlowSolution'])

  tree_info = {
               'bc_names': { 
                   'EdgeCenter' : [],
                   'FaceCenter' : ['bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6']
                   },
               'field_names' : { 'FlowSolution' : ['Zeros', 'Range'] },
              }

  comm.barrier()
  dist_tree = meshb_converter.meshb_to_cgns(files, tree_info, comm)

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert PT.Zone.n_vtx(zone) == 1331 and PT.Zone.n_cell(zone) == 5000

  vtx_distri = PT.maia.getDistribution(zone, 'Vertex')[1]
  sol = PT.get_node_from_path(zone, 'FlowSolution/Range')[1]
  assert (sol == np.arange(vtx_distri[0], vtx_distri[1])).all()

  # TODO BCs are poorly distributed
  bc = PT.get_node_from_name(zone, 'bc3')
  assert PT.maia.getDistribution(bc, 'Index')[1][2] == 200
