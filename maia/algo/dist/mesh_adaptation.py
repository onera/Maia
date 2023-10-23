import copy, time
import mpi4py.MPI as MPI

import maia
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
import maia.pytree        as PT
import maia.utils.logging as mlog
from   maia.utils         import np_utils
from maia.factory import full_to_dist
from maia.io.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info
from maia.algo.dist.adaptation_utils import duplicate_periodic_patch,\
                                            retrieve_initial_domain

from maia.algo.dist.merge_ids import merge_distributed_ids
from maia.transfer import protocols as EP

import numpy as np

import subprocess

from pathlib import Path


# TMP directory
tmp_repo   = Path('TMP_adapt_repo')

# INPUT files
in_file_meshb = tmp_repo / 'mesh.mesh'
in_file_solb  = tmp_repo / 'metric.sol'
in_file_fldb  = tmp_repo / 'field.sol'
# in_files = {'mesh': 'TMP_adapt_repo/mesh.mesh',
in_files = {'mesh': in_file_meshb,
            'sol' : in_file_solb ,
            'fld' : in_file_fldb }

mesh_back_file = tmp_repo / 'mesh_back.mesh'

# OUTPUT files
out_file_meshb = tmp_repo / 'mesh.o.mesh'
out_file_solb  = tmp_repo / 'mesh.o.sol'
out_file_fldb  = tmp_repo / 'field.itp.sol'
out_files = {'mesh': out_file_meshb,
             'sol' : out_file_solb ,
             'fld' : out_file_fldb }

# Feflo files arguments
feflo_args    = { 'isotrop'  : f"-iso               ".split(),
                  'from_fld' : f"-sol {in_file_solb}".split(),
                  'from_hess': f"-met {in_file_solb}".split()
}


def unpack_metric(dist_tree, metric_paths):
  """
  Unpacks the `metric` argument from `mesh_adapt` function.
  Assert no invalid path or argument is given and that paths leads to one or six fields.
  """

  if metric_paths is None:
    return list()

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]

  # > Get metrics nodes described by path
  if isinstance(metric_paths, str):
    container_name, fld_name = metric_paths.split('/')

    metric_nodes = [PT.get_node_from_path(zone, f"{container_name}/{fld_name}{suffix}") \
            for suffix in ['', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']]
    metric_nodes = [node for node in metric_nodes if node is not None] # Above list contains found node or None

  elif isinstance(metric_paths, list):
    assert len(metric_paths)==6, f"metric argument must be a str path or a list of 6 paths"
    metric_nodes = list()
    for path in metric_paths:
      metric_nodes.append(PT.get_node_from_path(zone, path))

  else:
    raise ValueError(f"Incorrect metric type (expected list or str).")


  # > Assert that metric is one or six fields
  if len(metric_nodes)==0:
    raise ValueError(f"Metric path \"{metric_paths}\" is invalid.")
  if len(metric_nodes)==7:
    raise ValueError(f"Metric path \"{metric_paths}\" simultaneously leads to scalar *and* tensor fields")
  if len(metric_nodes)!=1 and len(metric_nodes)!=6:
    raise ValueError(f"Metric path \"{metric_paths}\" leads to {len(metric_nodes)} nodes (1 or 6 expected).")


  return metric_nodes



def adapt_mesh_with_feflo(dist_tree, metric, comm, container_names=[], constraints=None, feflo_opts=""):
  """Run a mesh adaptation step using *Feflo.a* software.

  Important:
    - Feflo.a is an Inria software which must be installed by you and exposed in your ``$PATH``.
    - This API is experimental. It may change in the future.

  Input tree must be unstructured and have an element connectivity.
  Boundary conditions other than Vertex located are managed.

  Adapted mesh is returned as an independant distributed tree.

  **Setting the metric**

  Metric choice is available through the ``metric`` argument, which can take the following values:

  - *None* : isotropic adaptation is performed
  - *str* : path (starting a Zone_t level) to a scalar or tensorial vertex located field:

    - If the path leads to a scalar field (e.g. FlowSolution/Pressure), a metric is computed by
      Feflo from this field.
    - If the path leads to a tensorial field (e.g. FlowSolution/HessMach), we collect its 6 component (named
      after CGNS tensor conventions) and pass it to
      Feflo as a user-defined metric.

    ::

      FlowSolution FlowSolution_t
      ├───GridLocation GridLocation_t "Vertex"
      ├───Pressure DataArray_t R8 (100,)
      ├───HessMachXX DataArray_t R8 (100,)
      ├───HessMachYY DataArray_t R8 (100,)
      ├───...
      └───HessMachYZ DataArray_t R8 (100,)

  - *list of 6 str* : each string must be a path to a vertex located field representing one component
    of the user-defined metric tensor (expected order is ``XX, XY, XZ, YY, YZ, ZZ``)


  Args:
    dist_tree      (CGNSTree)    : Distributed tree to be adapted. Only U-Elements
      single zone trees are managed.
    metric         (str or list) : Path(s) to metric fields (see above)
    comm           (MPIComm)     : MPI communicator
    container_names(list of str) : Name of some Vertex located FlowSolution to project on the adapted mesh
    constraints    (list of str) : BC names of entities that must not be adapted (default to None)
    feflo_opts (str)             : Additional arguments passed to Feflo
  Returns:
    CGNSTree: Adapted mesh (distributed)

  Warning:
    Although this function interface is parallel, keep in mind that Feflo.a is a sequential tool.
    Input tree is thus internally gathered to a single process, which can cause memory issues on large cases.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #adapt_with_feflo@start
        :end-before: #adapt_with_feflo@end
        :dedent: 2
  """

  tmp_repo.mkdir(exist_ok=True)

  # > Get metric nodes
  metric_nodes = unpack_metric(dist_tree, metric)
  metric_names = PT.get_names(metric_nodes)

  n_metric_path = len(metric_nodes)
  metric_type = {0: 'isotrop', 1: 'from_fld', 6: 'from_hess'}[len(metric_nodes)]

  # > Get tree structure and names
  tree_info = get_tree_info(dist_tree, container_names)
  input_base = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  input_zone = PT.get_child_from_label(input_base, 'Zone_t')


  # > Gathering dist_tree on proc 0
  maia.algo.dist.redistribute_tree(dist_tree, 'gather.0', comm) # Modifie le dist_tree 


  # > CGNS to meshb conversion
  dicttag_to_bcinfo = list()

  if comm.Get_rank()==0:
    constraint_tags = cgns_to_meshb(dist_tree, in_files, metric_nodes, container_names, constraints)

    # Adapt with feflo
    feflo_itp_args = f'-itp {in_file_fldb}'.split() if len(container_names)!=0 else []
    feflo_command  = ['feflo.a', '-in', str(in_files['mesh'])] + feflo_args[metric_type] + feflo_itp_args + feflo_opts.split()        
    if len(constraint_tags['FaceCenter'])!=0:
      feflo_command  = feflo_command + ['-adap-surf-ids'] + [','.join(constraint_tags['FaceCenter'])]#[str(tag) for tag in constraint_tags['FaceCenter']]
    if len(constraint_tags['EdgeCenter'])!=0:
      feflo_command  = feflo_command + ['-adap-line-ids'] + [','.join(constraint_tags['EdgeCenter'])]#[str(tag) for tag in constraint_tags['EdgeCenter']]
    feflo_command  = ' '.join(feflo_command) # Split + join to remove useless spaces

    mlog.info(f"Start mesh adaptation using Feflo...")
    start = time.time()
    
    subprocess.run(feflo_command, shell=True)

    end = time.time()
    mlog.info(f"Feflo mesh adaptation completed ({end-start:.2f} s)")


  # > Recover original dist_tree
  maia.algo.dist.redistribute_tree(dist_tree, 'uniform', comm)


  # > Get adapted dist_tree
  adapted_dist_tree = meshb_to_cgns(out_files, tree_info, comm)

  # > Set names and copy base data
  adapted_base = PT.get_child_from_label(adapted_dist_tree, 'CGNSBase_t')
  adapted_zone = PT.get_child_from_label(adapted_base, 'Zone_t')
  PT.set_name(adapted_base, PT.get_name(input_base))
  PT.set_name(adapted_zone, PT.get_name(input_zone))

  to_copy = lambda n: PT.get_label(n) in ['Family_t']
  for node in PT.get_nodes_from_predicate(input_base, to_copy):
    PT.add_child(adapted_base, node)
  bc_n = PT.get_node_from_name(adapted_dist_tree, 'feflo_edge_bc_0')
  PT.new_node('FamilyName', label='FamilyName_t', value='BCS', parent=bc_n)
  maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/adapted1.cgns', comm)
  # > Copy BC data
  to_copy = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']
  for bc_path in PT.predicates_to_paths(adapted_zone, 'ZoneBC_t/BC_t'):
    adapted_bc = PT.get_node_from_path(adapted_zone, bc_path)
    input_bc   = PT.get_node_from_path(input_zone, bc_path)
    # print(f'bc_name = {PT.get_name(adapted_bc)}')
    if input_bc is not None:
      # print(f'  ---> DOING')
      PT.set_value(adapted_bc, PT.get_value(input_bc))
      for node in PT.get_nodes_from_predicate(input_bc, to_copy):
        PT.add_child(adapted_bc, node)


  return adapted_dist_tree



def periodic_adapt_mesh_with_feflo(dist_tree, metric, gc_paths, periodic, comm, container_names=[], feflo_opts="", **options):
  '''
    ROUTINE
    TODO
      travailler sur une copie du dist_tree
      accepter maillage avec plusieurs noeuds elements de meme dim
        toujours n'avoir qu'un noeud element/dim (l'imposer dans la doc + assert ?)
      figer certaines parties du volume
      inclure rm_invalid elements dans update_elt_numbering
      se passer de la vision point_list au profit des tags ??
      gerer les rotations
      mutualiser avec `adapt_mesh_with_feflo`

      passer le tmp_repo en arguement
  '''


  mlog.info(f"\n\n[Periodic adaptation] Step #1: Duplicating periodic patch...")
  pdist_tree = copy.deepcopy(dist_tree)


  # > Get matching vtx
  dist_base = PT.get_child_from_label(pdist_tree, 'CGNSBase_t')
  zone_bc_n = PT.get_node_from_label(pdist_tree, 'ZoneBC_t')
  # PT.new_Family('GC_TO_CONVERT1', parent=dist_base)

  for i_gc, gc_path in enumerate(gc_paths):
    gc_n    = PT.get_node_from_path(pdist_tree, gc_path)
    gc_name = PT.get_name(gc_n)
    gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
    gc_loc  = PT.Subset.GridLocation(gc_n)
    gc_distrib_n = PT.maia.getDistribution(gc_n)
    bc_n = PT.new_BC(name=gc_name,
                     type='FamilySpecified',
                     point_list=gc_pl,
                     loc=gc_loc,
                     family=f'GC_TO_CONVERT_{i_gc}',
                     parent=zone_bc_n)
    PT.add_child(bc_n, gc_distrib_n)

  maia.algo.dist.connect_1to1_families(pdist_tree, ('GC_TO_CONVERT_0', 'GC_TO_CONVERT_1'), comm, periodic=periodic, location='Vertex')


  maia.algo.dist.redistribute_tree(pdist_tree, 'gather.0', comm) # Modifie le dist_tree 
  PT.rm_nodes_from_name(pdist_tree, ':CGNS#Distribution')
  
  gc_name = gc_paths[0].split('/')[-1]
  periodic_values, new_vtx_num, bcs_to_update, bcs_to_retrieve = \
    duplicate_periodic_patch(pdist_tree, gc_name, comm)
  pdist_tree = full_to_dist.full_to_dist_tree(pdist_tree, comm)

  # print(f'new_vtx_num = {new_vtx_num}')

  # PT.print_tree(pdist_tree)
  is_cell_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='CellCenter'
  is_face_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='FaceCenter'
  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='EdgeCenter'
  is_vtx_bc  = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='Vertex'

  # PT.rm_nodes_from_predicate(dist_tree, is_cell_bc)
  # PT.rm_nodes_from_predicate(pdist_tree, is_face_bc)
  # PT.rm_nodes_from_predicate(pdist_tree, is_edge_bc)
  # PT.rm_nodes_from_predicate(pdist_tree, is_vtx_bc)
  
  # PT.rm_nodes_from_name(pdist_tree, 'BAR_2.0')
  # PT.rm_nodes_from_name(pdist_tree, 'TRI_3.0')
  # PT.rm_nodes_from_name(pdist_tree, 'ridge*')
  
  bc_names = list()
  zone_bc_n = PT.get_node_from_label(pdist_tree, 'ZoneBC_t')
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_edge_bc):
    bc_names.append(PT.get_name(bc_n))
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'FaceCenter')

  maia.io.dist_tree_to_file(pdist_tree, 'OUTPUT/extended_domain.cgns', comm)
  # sys.exit()

  for bc_name in bc_names:
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'EdgeCenter')


  mlog.info(f"\n\n[Periodic adaptation] Step #2: First adaptation constraining periodic patches boundaries...")
  padapted_dist_tree = adapt_mesh_with_feflo( pdist_tree, metric, comm,
                                              container_names=container_names,
                                              constraints=['tri_3_periodic', 'tri_3_constraint'],
                                              feflo_opts=feflo_opts)
  padapted_dist_base = PT.get_child_from_label(padapted_dist_tree, 'CGNSBase_t')
  PT.new_Family('BCS', parent=padapted_dist_base)

  bc_names = list()
  zone_bc_n = PT.get_node_from_label(padapted_dist_tree, 'ZoneBC_t')
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_edge_bc):
    bc_names.append(PT.get_name(bc_n))
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'FaceCenter')


  zone = PT.get_node_from_label(padapted_dist_tree, 'Zone_t')
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  bc_constraint_n = PT.get_child_from_name(zone_bc_n, 'tetra_4_constraint')
  bc_periodic_n   = PT.get_child_from_name(zone_bc_n, 'tetra_4_periodic')
  bc_pl_constraint = PT.get_value(PT.get_child_from_name(bc_constraint_n, 'PointList'))[0]
  bc_pl_periodic   = PT.get_value(PT.get_child_from_name(bc_periodic_n  , 'PointList'))[0]

  n_cell = PT.Zone.n_cell(zone)
  fld = np.zeros(n_cell, dtype=np.float64)
  fld[bc_pl_constraint  -1] = 2.
  fld[bc_pl_periodic-1] = 1.
  PT.new_FlowSolution('FSolution', fields={'tag_face':fld}, loc='CellCenter', parent=zone)
  # PT.rm_nodes_from_name(dist_tree, 'ZoneGridConnectivity')

  PT.rm_nodes_from_name(padapted_dist_tree, 'Metric')

  maia.io.dist_tree_to_file(padapted_dist_tree, 'OUTPUT/first_adaptation.cgns', comm)

  for bc_name in bc_names:
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'EdgeCenter')

  # sys.exit()


  mlog.info(f"\n\n[Periodic adaptation] #3: Removing initial domain...")
  adapted_dist_tree = copy.deepcopy(padapted_dist_tree)
  maia.algo.dist.redistribute_tree(adapted_dist_tree, 'gather.0', comm) # Modifie le dist_tree 
  PT.rm_nodes_from_name(adapted_dist_tree, ':CGNS#Distribution')
  # print(f'n_vtx = {PT.Zone.n_vtx(PT.get_node_from_label(adapted_dist_tree, "Zone_t"))}')

  retrieve_initial_domain(adapted_dist_tree, gc_name, periodic_values, new_vtx_num,\
                          bcs_to_update, bcs_to_retrieve, comm)
  adapted_dist_tree = full_to_dist.full_to_dist_tree(adapted_dist_tree, comm)

  bc_names = list()
  zone_bc_n = PT.get_node_from_label(adapted_dist_tree, 'ZoneBC_t')
  for bc_n in PT.get_nodes_from_predicate(zone_bc_n, is_edge_bc):
    bc_names.append(PT.get_name(bc_n))
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'FaceCenter')

  maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/initial_domain.cgns', comm)
  
  for bc_name in bc_names:
    bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
    PT.set_value(PT.get_child_from_name(bc_n, 'GridLocation'), 'EdgeCenter')

  sys.exit()


  mlog.info(f"\n\n[Periodic adaptation] #4: Perform last adaptation constraining periodicities...")
  fadapted_dist_tree = adapt_mesh_with_feflo( adapted_dist_tree, metric, comm,
                                              container_names=container_names,
                                              constraints=[gc_name, 'Xmax'],
                                              feflo_opts=feflo_opts)
  
  maia.io.dist_tree_to_file(fadapted_dist_tree, 'OUTPUT/adapted.cgns', comm)

  return fadapted_dist_tree
