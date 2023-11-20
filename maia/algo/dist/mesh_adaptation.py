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
                                            retrieve_initial_domain,\
                                            rm_feflo_added_elt

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


def get_gc_path_pairs(tree):
  # > Check type of GCs in tree (maybe not necessary)
  is_gc = lambda n : PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t']
  for gc_n in PT.get_nodes_from_predicate(tree, is_gc):
    assert PT.GridConnectivity.is1to1(gc_n) and PT.GridConnectivity.isperiodic(gc_n),\
           'Tree with not periodic or 1o1 joins are not managed.'
  
  # > Get paths of GC pairs
  is_periodic_1to1 = lambda n: PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t'] and\
                               PT.GridConnectivity.is1to1(n) and\
                               PT.GridConnectivity.isperiodic(n)and\
                               PT.Subset.GridLocation(n)=='FaceCenter'
  query = ["CGNSBase_t", "Zone_t", "ZoneGridConnectivity_t", is_periodic_1to1]
  per_1to1_gc_paths = PT.predicates_to_paths(tree, query)
  treated_gcs = list()
  gc_paths = (list(),list())
  periodic_values = (list(),list())
  for gc_path in per_1to1_gc_paths:
    if gc_path not in treated_gcs:
      gc_n = PT.get_node_from_path(tree, gc_path)
      gc_donor_name_n = PT.get_child_from_name(gc_n, 'GridConnectivityDonorName')
      assert gc_donor_name_n is not None, 'Joins must have GridConnectivityDonorName node.'
      gc_donor_name = PT.get_value(gc_donor_name_n)
      gc_donor_path = '/'.join(gc_path.split('/')[:-1]+[gc_donor_name])
      gc_paths[0].append(gc_path)
      gc_paths[1].append(gc_donor_path)
      treated_gcs.append(gc_donor_path)

      donor_gc_n = PT.get_node_from_path(tree, gc_donor_path)
      periodic_values[0].append(PT.GridConnectivity.periodic_dict(gc_n))
      periodic_values[1].append(PT.GridConnectivity.periodic_dict(donor_gc_n))

  return gc_paths, periodic_values


def store_gcs_as_bcs(tree, gc_paths):
  zone_bc_n = PT.get_node_from_label(tree, 'ZoneBC_t')
  periodic_values = (list(), list())
  for i_side, side_gc_paths in enumerate(gc_paths):
    for i_gc, gc_path in enumerate(side_gc_paths):
      gc_n    = PT.get_node_from_path(tree, gc_path)
      gc_name = PT.get_name(gc_n)
      gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
      gc_loc  = PT.Subset.GridLocation(gc_n)
      assert gc_loc=='FaceCenter', ''
      gc_distrib = PT.get_value(PT.maia.getDistribution(gc_n, 'Index'))
      bc_n = PT.new_BC(name=gc_name,
                       type='FamilySpecified',
                       point_list=gc_pl,
                       loc=gc_loc,
                       parent=zone_bc_n)
      PT.maia.newDistribution({'Index':gc_distrib}, parent=bc_n)


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

  Periodic mesh adaptation is available by activating the ``periodic`` argument. Informations from GridConnectivity_t nodes 
  in dist_tree will be used to perform mesh adaptation. Only periodic 1to1 GridConnectivities are managed.

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
    '''
    1ere extension de domaine:
      - tag des cellules qui ont un vtx dans la gc
      - création de la surface de contrainte:
        - connectivité des cellules tagguées sans les vertex de la gc
        - toutes les éléments qui ne sont pas dans la connectivité des BCs
      - duplication des éléments de la surface contrainte
      - déplacement des vertex du patch
      - merge des surfaces de la GC

    2eme extension de domaine:
      - get bc cellules (issu du mesh adapté)
      - duplication des éléments de la gc
      - déplacement des vertex du patch
      - merge des surfaces de contrainte


    Issues:
       - would ngon be faster ?
       - fix coarse ls89 with face on blade
       - how to back mesh ?
       - retrieve ridges that has been deleted during domain extension

    TODO:
       - manage n_range of cells
       - improve perfos: face in cells detection (with ngon ?)
       - ne plus avoir besoin des GCs FaceCenter
       - gérer la périodisation des champs (rotation)
    '''
    start = time.time()
    adapted_dist_tree = copy.deepcopy(dist_tree) # TODO: shallow_copy sufficient ?


    # > Get periodic infos + prepare cgns
    gc_paths, periodic_values = get_gc_path_pairs(adapted_dist_tree)
    store_gcs_as_bcs(adapted_dist_tree, gc_paths)


    mlog.info(f"[Periodic adaptation] Step #1: Duplicating periodic patch...")
    maia.algo.dist.redistribute_tree(adapted_dist_tree, 'gather.0', comm) # Modifie le dist_tree 
    PT.rm_nodes_from_name(adapted_dist_tree, ':CGNS#Distribution')

    bcs_to_constrain = list()
    if comm.rank==0:
      new_vtx_num, bcs_to_constrain, bcs_to_update, bcs_to_retrieve = \
        duplicate_periodic_patch(adapted_dist_tree, gc_paths, periodic_values)
    adapted_dist_tree = full_to_dist.full_to_dist_tree(adapted_dist_tree, comm, owner=0)
    bcs_to_constrain = comm.bcast(bcs_to_constrain, root=0)

    end = time.time()
    # maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/extended_domain.cgns', comm)
    mlog.info(f"[Periodic adaptation] Step #1 completed: ({end-start:.2f} s)")


    mlog.info(f"[Periodic adaptation] Step #2: First adaptation constraining periodic patches boundaries...")
    adapted_dist_tree = _adapt_mesh_with_feflo( adapted_dist_tree, metric, comm,
                                                container_names,
                                                bcs_to_constrain,
                                                feflo_opts)
    padapted_dist_base = PT.get_child_from_label(adapted_dist_tree, 'CGNSBase_t')
    # maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/first_adaptation.cgns', comm)


    mlog.info(f"[Periodic adaptation] #3: Removing initial domain...")
    start = time.time()
    maia.algo.dist.redistribute_tree(adapted_dist_tree, 'gather.0', comm) # Modifie le dist_tree 
    PT.rm_nodes_from_name(adapted_dist_tree, ':CGNS#Distribution')

    if comm.rank==0:
      retrieve_initial_domain(adapted_dist_tree, gc_paths, periodic_values, new_vtx_num,\
                              bcs_to_update, bcs_to_retrieve)
    adapted_dist_tree = full_to_dist.full_to_dist_tree(adapted_dist_tree, comm, owner=0)

    end = time.time()
    # maia.io.dist_tree_to_file(adapted_dist_tree, 'OUTPUT/initial_domain.cgns', comm)
    mlog.info(f"[Periodic adaptation] Step #3 completed: ({end-start:.2f} s)")
    # sys.exit()


    mlog.info(f"[Periodic adaptation] #4: Perform last adaptation constraining periodicities...")
    gc_constraints = list()
    for i_side, side_gc_paths in enumerate(gc_paths):
      for i_gc, gc_path in enumerate(side_gc_paths):
        gc_constraints.append(gc_path.split('/')[-1])
    adapted_dist_tree = _adapt_mesh_with_feflo( adapted_dist_tree, metric, comm,
                                                container_names,
                                                gc_constraints,
                                                feflo_opts)
    

    # > Retrieve periodicities + cleaning file
    PT.rm_nodes_from_name_and_label(adapted_dist_tree, 'PERIODIC', 'Family_t')
    PT.rm_nodes_from_name_and_label(adapted_dist_tree, 'GCS',      'Family_t')
    PT.rm_nodes_from_name_and_label(adapted_dist_tree, 'maia_topo','FlowSolution_t')
    PT.rm_nodes_from_name_and_label(adapted_dist_tree, 'tetra_4_periodic*','BC_t')

    zone_bc_n = PT.get_node_from_label(adapted_dist_tree, 'ZoneBC_t')
    bc_nodes = list()
    n_interface = len(gc_paths[0])
    for i_side, side_gc_paths in enumerate(gc_paths):
      for i_gc, gc_path in enumerate(side_gc_paths):
        bc_name = gc_path.split('/')[-1]
        bc_n = PT.get_node_from_name(adapted_dist_tree, bc_name, 'BC_t')
        PT.rm_children_from_label(bc_n, 'FamilyName_t')
        PT.new_node('FamilyName', label='FamilyName_t', value=f'BC_TO_CONVERT_{i_gc}_{i_side}', parent=bc_n)
        bc_nodes.append(bc_n)

    periodic_names = ["rotation_center", "rotation_angle", "translation"]
    for i_interface in range(n_interface):
      maia.algo.dist.connect_1to1_families(adapted_dist_tree, (f'BC_TO_CONVERT_{i_interface}_0', f'BC_TO_CONVERT_{i_interface}_1'), comm, periodic=periodic_values[0][i_interface])

    for i_side, side_gc_paths in enumerate(gc_paths):
      for i_gc, gc_path in enumerate(side_gc_paths):
        gc_name = gc_path.split('/')[-1]+'_0'
        gc_n = PT.get_node_from_name(adapted_dist_tree, gc_name, 'GridConnectivity_t')
        gcd_name_n = PT.get_child_from_name(gc_n, 'GridConnectivityDonorName')
        PT.set_value(gcd_name_n, PT.get_value(gcd_name_n)[:-2])
        PT.rm_children_from_label(gc_n, 'FamilyName_t')
        PT.set_name(gc_n, gc_path.split('/')[-1])

    for bc_n in bc_nodes:
      PT.add_child(zone_bc_n, bc_n)

    for i_interface in range(n_interface):
      maia.algo.dist.connect_1to1_families(adapted_dist_tree, (f'BC_TO_CONVERT_{i_interface}_0', f'BC_TO_CONVERT_{i_interface}_1'), comm, periodic=periodic_values[0][i_interface], location='Vertex')

    for i_side, side_gc_paths in enumerate(gc_paths):
      for i_gc, gc_path in enumerate(side_gc_paths):
        gc_name = gc_path.split('/')[-1]+'_0'
        gc_n = PT.get_node_from_name(adapted_dist_tree, gc_name, 'GridConnectivity_t')
        PT.rm_children_from_label(gc_n, 'FamilyName_t')


    maia.algo.dist.redistribute_tree(adapted_dist_tree, 'gather.0', comm) # Modifie le dist_tree 
    PT.rm_nodes_from_name(adapted_dist_tree, ':CGNS#Distribution')
    if comm.rank==0:
      zone = PT.get_node_from_label(adapted_dist_tree, 'Zone_t')
      rm_feflo_added_elt(zone)
    adapted_dist_tree = full_to_dist.full_to_dist_tree(adapted_dist_tree, comm, owner=0)


  else:
    adapted_dist_tree = adapt_mesh_with_feflo(dist_tree, metric, comm, container_names, constraints, feflo_opts)
  
  return adapted_dist_tree
