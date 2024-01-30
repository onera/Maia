import time
import subprocess
from pathlib import Path

import maia
import maia.pytree        as PT
import maia.utils.logging as mlog

from maia.io.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info
from maia.algo.dist.matching_jns_tools import add_joins_donor_name, get_matching_jns
from maia.algo.dist.adaptation_utils import convert_vtx_gcs_as_face_bcs,\
                                            deplace_periodic_patch,\
                                            retrieve_initial_domain,\
                                            rm_feflo_added_elt


# TMP directory
tmp_repo   = Path('TMP_adapt_repo')

# INPUT files
in_file_meshb = tmp_repo / 'mesh.mesh'
in_file_solb  = tmp_repo / 'metric.sol'
in_file_fldb  = tmp_repo / 'field.sol'
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


def _adapt_mesh_with_feflo(dist_tree, metric, comm, container_names, constraints, feflo_opts):
  tmp_repo.mkdir(exist_ok=True)

  # > Get metric nodes
  metric_nodes = unpack_metric(dist_tree, metric)
  metric_type = {0: 'isotrop', 1: 'from_fld', 6: 'from_hess'}[len(metric_nodes)]

  # > Get tree structure and names
  tree_info = get_tree_info(dist_tree, container_names)
  tree_info = comm.bcast(tree_info, root=0)
  input_base = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  input_zone = PT.get_child_from_label(input_base, 'Zone_t')


  # > CGNS to meshb conversion
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

  # > Copy BC data
  to_copy = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']
  for bc_path in PT.predicates_to_paths(adapted_zone, 'ZoneBC_t/BC_t'):
    adapted_bc = PT.get_node_from_path(adapted_zone, bc_path)
    input_bc   = PT.get_node_from_path(input_zone, bc_path)
    if input_bc is not None:
      PT.set_value(adapted_bc, PT.get_value(input_bc))
      for node in PT.get_nodes_from_predicate(input_bc, to_copy):
        PT.add_child(adapted_bc, node)

  return adapted_dist_tree

def _adapt_mesh_with_feflo_perio(dist_tree, metric, comm, container_names, feflo_opts):
  '''
  Assume that : 
    - Only one Element node for each dimension
    - Mesh is full TRI_3 and TETRA_4
    - GridConnectivities have no common vertices


  1ere extension de domaine:
    1/ tag des cellules qui ont un vtx dans la gc
    2/ création de la surface de contrainte:
      - création des faces à partir de la connectivité des cellules tagguées sans les vertex de la gc
      - suppression des faces qui sont déjà définies dans les BCs, ou les possibles doublons de face créées par 2 cellules
    3/ duplication des éléments de la surface contrainte
    4/ déplacement des vertex du patch
    5/ merge des surfaces de la GC

  2eme extension de domaine:
    1/ get bc cellules (issu du mesh adapté)
    2/ duplication des éléments de la gc
    3/ déplacement des vertex du patch
    4/ merge des surfaces de contrainte
    5/ retrouver les ridges (et corners) disparus à l'étape 1


  Issues:
    - would ngon be faster ?
    - gérer la périodisation des champs (rotation)
        - [x] vector
        - [ ] tensor -> useful for metric
        - [?] some variables (angle in torus case)
    - cas du cone -> arete sur l'axe bien gérée ?

  TODO:
    - manage back mesh
        [!] beware of BC ordering that have been changed
    - manage n_range of cells
    - tout adapter pour le 2d ?
  '''
  tree = PT.deep_copy(dist_tree) # Do not modify input tree 

  start = time.time()
  # > Get periodic infos
  add_joins_donor_name(tree, comm) # Add missing joins donor names
  perio_jns_pairs = get_matching_jns(tree, lambda n : PT.GridConnectivity.isperiodic(n))
  jn_pairs_and_values = dict()
  for pair in perio_jns_pairs:
    gc_nodes = (PT.get_node_from_path(tree, gc_path) for gc_path in pair)
    jn_pairs_and_values[pair] = [PT.GridConnectivity.periodic_values(gc) for gc in gc_nodes]

  convert_vtx_gcs_as_face_bcs(tree, comm)

  mlog.info(f"[Periodic adaptation] Step #1: Duplicating periodic patch...")

  new_vtx_num, bcs_to_constrain, bcs_to_retrieve = deplace_periodic_patch(tree, perio_jns_pairs, comm)

  end = time.time()
  # maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/extended_domain.cgns', comm)
  mlog.info(f"[Periodic adaptation] Step #1 completed: ({end-start:.2f} s)")


  mlog.info(f"[Periodic adaptation] Step #2: First adaptation constraining periodic patches boundaries...")
  maia.algo.dist.redistribute_tree(tree, 'gather.0', comm)
  tree = _adapt_mesh_with_feflo(tree, metric, comm, container_names, bcs_to_constrain, feflo_opts)


  mlog.info(f"[Periodic adaptation] #3: Removing initial domain...")
  start = time.time()

  retrieve_initial_domain(tree, jn_pairs_and_values, new_vtx_num, bcs_to_retrieve, comm)

  end = time.time()
  # maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/initial_domain.cgns', comm)
  mlog.info(f"[Periodic adaptation] Step #3 completed: ({end-start:.2f} s)")


  mlog.info(f"[Periodic adaptation] #4: Perform last adaptation constraining periodicities...")
  gc_constraints = [PT.path_tail(gc_path) for pair in perio_jns_pairs for gc_path in pair]
  maia.algo.dist.redistribute_tree(tree, 'gather.0', comm)
  tree = _adapt_mesh_with_feflo(tree, metric, comm, container_names, gc_constraints, feflo_opts)


  # > Retrieve periodicities + cleaning file
  PT.rm_nodes_from_name_and_label(tree, 'PERIODIC', 'Family_t', depth=2)
  PT.rm_nodes_from_name_and_label(tree, 'GCS',      'Family_t', depth=2)
  for zone in PT.get_all_Zone_t(tree):
    PT.rm_children_from_name_and_label(zone, 'maia_topo','FlowSolution_t')
    PT.rm_nodes_from_name_and_label(zone, 'tetra_4_periodic*','BC_t', depth=2)

  # > Set family name in BCs for connect_match
  for i_jn, jn_pair in enumerate(perio_jns_pairs):
    for i_gc, gc_path in enumerate(jn_pair):
      bc_path = PT.update_path_elt(gc_path, 2, lambda n: 'ZoneBC') # gc has been stored as a BC
      bc_n = PT.get_node_from_path(tree, bc_path)
      PT.update_child(bc_n, 'FamilyName', 'FamilyName_t', f'BC_TO_CONVERT_{i_jn}_{i_gc}')

  for i_jn, jn_values in enumerate(jn_pairs_and_values.values()):
    maia.algo.dist.connect_1to1_families(tree, (f'BC_TO_CONVERT_{i_jn}_0', f'BC_TO_CONVERT_{i_jn}_1'), comm, 
                                        periodic=jn_values[0].asdict(True), location='Vertex')

  # > Remove '_0' in the created GCs names
  for jn_pair in perio_jns_pairs:
    for gc_path in jn_pair:
      gc_n = PT.get_node_from_path(tree, gc_path+'_0')
      PT.set_name(gc_n, PT.path_tail(gc_path))
      gcd_name_n = PT.get_child_from_name(gc_n, 'GridConnectivityDonorName')
      PT.set_value(gcd_name_n, PT.get_value(gcd_name_n)[:-2])
      PT.rm_children_from_label(gc_n, 'FamilyName_t')

  rm_feflo_added_elt(PT.get_node_from_label(tree, 'Zone_t'), comm)

  return tree



def adapt_mesh_with_feflo(dist_tree, metric, comm, container_names=[], constraints=None, periodic=False, feflo_opts=""):
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

  **Periodic mesh adaptation**

  Periodic mesh adaptation is available by activating the ``periodic`` argument. Information from 
  periodic 1to1 GridConnectivity_t nodes in dist_tree will be used to perform mesh adaptation.

  Args:
    dist_tree      (CGNSTree)    : Distributed tree to be adapted. Only U-Elements
      single zone trees are managed.
    metric         (str or list) : Path(s) to metric fields (see above)
    comm           (MPIComm)     : MPI communicator
    container_names(list of str) : Name of some Vertex located FlowSolution to project on the adapted mesh
    constraints    (list of str) : BC names of entities that must not be adapted (default to None)
    periodic       (boolean)     : perform periodic mesh adaptation
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

  if periodic:
    adapted_dist_tree = _adapt_mesh_with_feflo_perio(dist_tree, metric, comm, container_names, feflo_opts)
  else:
    # > Gathering dist_tree on proc 0
    maia.algo.dist.redistribute_tree(dist_tree, 'gather.0', comm) # Modifie le dist_tree

    adapted_dist_tree = _adapt_mesh_with_feflo(dist_tree, metric, comm, container_names, constraints, feflo_opts)
    PT.rm_nodes_from_name_and_label(adapted_dist_tree, 'maia_topo','FlowSolution_t')

    # > Recover original dist_tree
    maia.algo.dist.redistribute_tree(dist_tree, 'uniform', comm)

  return adapted_dist_tree
