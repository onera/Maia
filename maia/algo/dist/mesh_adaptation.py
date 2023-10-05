import copy, time
import mpi4py.MPI as MPI

import maia
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
import maia.pytree        as PT
import maia.utils.logging as mlog
from   maia.utils         import np_utils
from maia.io.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info
from .adaptation_utils import elmt_pl_to_vtx_pl,\
                              tag_elmt_owning_vtx,\
                              add_periodic_elmt,\
                              add_constraint_bcs,\
                              update_elt_vtx_numbering,\
                              remove_elts_from_pl,\
                              apply_offset_to_elt,\
                              find_invalid_elts

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



def adapt_mesh_with_feflo(dist_tree, metric, comm, container_names=[], feflo_opts=""):
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
    cgns_to_meshb(dist_tree, in_files, metric_nodes, container_names)

    # Adapt with feflo
    feflo_itp_args = f'-itp {in_file_fldb}'.split() if len(container_names)!=0 else []
    feflo_command  = ['feflo.a', '-in', str(in_files['mesh'])] + feflo_args[metric_type] + feflo_itp_args + feflo_opts.split()        
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
    PT.set_value(adapted_bc, PT.get_value(input_bc))
    for node in PT.get_nodes_from_predicate(input_bc, to_copy):
      PT.add_child(adapted_bc, node)


  return adapted_dist_tree



def periodic_adapt_mesh_with_feflo(dist_tree, metric, comm, container_names=[], feflo_opts=""):

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


  # > First adaptation: one periodic side
  mlog.info(f"\n\n[Periodic adaptation] #1: Duplicating periodic patch...")

  pdist_tree = copy.deepcopy(dist_tree)
  dist_zone = PT.get_nodes_from_label(dist_tree, 'Zone_t')
  assert len(dist_zone)==1
  dist_zone = dist_zone[0]
  PT.Zone.get_ordered_elements_per_dim(dist_zone)

  # > Get GCs involved in periodicity and transform them into vtx pointlist
  gc_nodes = PT.get_nodes_from_label(pdist_tree, 'GridConnectivity_t')
  gc_names = [PT.get_name(n) for n in gc_nodes]
  assert len(gc_nodes)==2
  gc_n  = gc_nodes[0]
  assert PT.Subset.GridLocation(gc_n)=='EdgeCenter' # 2d for now
  translation = PT.get_value(PT.get_node_from_name(gc_n, 'Translation'))
  gc_pl  = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_pld = PT.get_value(PT.get_child_from_name(gc_n, 'PointListDonor'))[0]
  gc_vtx_pl  = elmt_pl_to_vtx_pl(dist_zone, gc_pl , 'BAR_2')
  gc_vtx_pld = elmt_pl_to_vtx_pl(dist_zone, gc_pld, 'BAR_2')

  # > Which cells are connected to donor_vtx ?
  n_tri = PT.Zone.n_cell(dist_zone)
  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)=='TRI_3'
  tri_nodes = PT.get_nodes_from_predicate(dist_zone, is_tri_elt)
  tri_conn  = [PT.get_value(PT.get_child_from_name(tri_n, 'ElementConnectivity')) for tri_n in tri_nodes]
  tri_conn  = np_utils.concatenate_np_arrays(tri_conn)[1]
  gc_tri_pl = tag_elmt_owning_vtx(n_tri, tri_conn, gc_vtx_pld)
  


  # > Create connectivity of periodic elements
  pdist_zone = PT.get_node_from_label(pdist_tree, 'Zone_t')
  
  zone_bc_n = PT.get_node_from_label(pdist_zone, 'ZoneBC_t')
  pl_vol  = np.arange(1, n_tri+1, dtype=np.int32)
  pl_vol  = np.delete(pl_vol, gc_tri_pl-1).reshape((1,-1), order='F')
  pl_volp = np.arange(n_tri, n_tri+gc_tri_pl.size+1, dtype=np.int32)
  pl_volp = pl_volp.reshape((1,-1), order='F')
  pl_volc = (gc_tri_pl).reshape((1,-1), order='F')
  PT.new_BC(name='vol',            type='BCWall', point_list=pl_vol , loc='FaceCenter', family='BCS', parent=zone_bc_n)
  PT.new_BC(name='vol_periodic',   type='BCWall', point_list=pl_volp, loc='FaceCenter', family='BCS', parent=zone_bc_n)
  PT.new_BC(name='vol_constraint', type='BCWall', point_list=pl_volc, loc='FaceCenter', family='BCS', parent=zone_bc_n)

  for gc_n in gc_nodes:
    gc_name = PT.get_name(gc_n)
    gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
    PT.new_BC(name=gc_name,
              type='BCWall',
              point_list=gc_pl,
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)
  
  n_vtx_toadd, new_num_vtx = add_periodic_elmt(pdist_zone, gc_tri_pl, gc_vtx_pl, gc_vtx_pld, translation, comm)
  

  # Create new BAR_2 elmts and associated BCs to constrain mesh adaptation
  add_constraint_bcs(pdist_zone, new_num_vtx)


  PT.rm_nodes_from_name(pdist_tree, 'ZoneGridConnectivity')
  maia.io.write_tree(pdist_tree, 'OUTPUT/square_extended.cgns')

  ptree_info = get_tree_info(pdist_tree, container_names)

  # sys.exit()
  '''
  ROUTINE
    etendre la partition
      transférer les flowsol
    figer la surface creee (tout a gauche) et la correspodant (tout a droite -1 rangée)
    reporter les modifs
      transférer les flowsol
    adapter le bloc final en figeant les BCs
  TODO
    travailler sur une copie du dist_tree
    accepter maillage avec plusieurs noeuds elements de meme dim
    figer certaines parties du volume
    toujours n'avoir qu'un noeud element/dim (l'imposer dans la doc + assert ?)
    inclure rm_invalid elements dans update_elt_numbering
    faut reporter les BCs periodisée !!
    se passer de la vision point_list au profit des tags ??
  '''

  mlog.info(f"\n\n[Periodic adaptation] #2: First adaptation constraining periodic patches boundaries...")

  if comm.Get_rank()==0:
    cgns_to_meshb(pdist_tree, in_files, metric_nodes, container_names)

    # Adapt with feflo
    feflo_itp_args = f'-itp {in_file_fldb}'.split() if len(container_names)!=0 else []
    feflo_command  = ['feflo.a', '-in', str(in_files['mesh'])] + feflo_args[metric_type] + feflo_itp_args + feflo_opts.split()        
    feflo_command  = ' '.join(feflo_command) # Split + join to remove useless spaces

    mlog.info(f"Start mesh adaptation using Feflo...")
    start = time.time()
    
    subprocess.run(feflo_command, shell=True)

    end = time.time()
    mlog.info(f"Feflo mesh adaptation completed ({end-start:.2f} s)")


  # > Get adapted dist_tree

  padapted_dist_tree = meshb_to_cgns(out_files, ptree_info, comm)
  # > Set names and copy base data
  padapted_base = PT.get_child_from_label(padapted_dist_tree, 'CGNSBase_t')
  padapted_zone = PT.get_child_from_label(padapted_base, 'Zone_t')
  PT.set_name(padapted_base, PT.get_name(input_base))
  PT.set_name(padapted_zone, PT.get_name(pdist_zone))

  to_copy = lambda n: PT.get_label(n) in ['Family_t']
  for node in PT.get_nodes_from_predicate(input_base, to_copy):
    PT.add_child(padapted_base, node)

  # > Copy BC data
  to_copy = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']
  for bc_path in PT.predicates_to_paths(padapted_zone, 'ZoneBC_t/BC_t'):
    padapted_bc = PT.get_node_from_path(padapted_zone, bc_path)
    input_bc   = PT.get_node_from_path(pdist_zone, bc_path)
    if input_bc is not None:
      PT.set_value(padapted_bc, PT.get_value(input_bc))
      for node in PT.get_nodes_from_predicate(input_bc, to_copy):
        PT.add_child(padapted_bc, node)

  maia.io.write_tree(padapted_dist_tree, 'OUTPUT/first_adaptation.cgns')






  mlog.info(f"\n\n[Periodic adaptation] #3: Removing initial periodic patch...")

  adapted_dist_tree = copy.deepcopy(padapted_dist_tree)
  adapted_dist_zone = PT.get_node_from_label(adapted_dist_tree, 'Zone_t')

  n_vtx = PT.Zone.n_vtx(adapted_dist_zone)
  n_elt = PT.Zone.n_cell(adapted_dist_zone)

  zone_bc_n = PT.get_child_from_label(adapted_dist_zone, 'ZoneBC_t')


  # > Removing old periodic patch
  bc_to_rm = PT.get_child_from_name(zone_bc_n, 'vol_constraint')
  bc_to_rm_pl = PT.get_value(PT.get_child_from_name(bc_to_rm, 'PointList'))[0]
  bc_to_rm_vtx_pl  = elmt_pl_to_vtx_pl(adapted_dist_zone, bc_to_rm_pl, 'TRI_3')
  n_elt_to_rm = bc_to_rm_pl.size

  bc_to_keep = PT.get_child_from_name(zone_bc_n, 'fixed')
  bc_to_keep_pl = PT.get_value(PT.get_child_from_name(bc_to_keep, 'PointList'))[0]
  bc_to_keep_vtx_pl = elmt_pl_to_vtx_pl(adapted_dist_zone, bc_to_keep_pl, 'BAR_2')
  n_vtx_to_keep = bc_to_keep_vtx_pl.size


  tag_vtx = np.isin(bc_to_rm_vtx_pl, bc_to_keep_vtx_pl) # True where vtx is 
  preserved_vtx_id = bc_to_rm_vtx_pl[tag_vtx][0]
  bc_to_rm_vtx_pl = bc_to_rm_vtx_pl[np.invert(tag_vtx)]
  n_vtx_to_rm = bc_to_rm_vtx_pl.size


  # Compute new vtx numbering

  vtx_tag_n = PT.get_node_from_name(adapted_dist_zone, 'vtx_tag')
  vtx_tag   = PT.get_value(vtx_tag_n)
  vtx_tag = np.delete(vtx_tag, bc_to_rm_vtx_pl-1)
  PT.set_value(vtx_tag_n, vtx_tag)

  bc_fixed_n = PT.get_node_from_name_and_label(adapted_dist_zone, 'fixed', 'BC_t')
  bc_fixed_pl = PT.get_value(PT.get_child_from_name(bc_fixed_n, 'PointList'))[0]
  bc_fixed_vtx_pl = elmt_pl_to_vtx_pl(adapted_dist_zone, bc_fixed_pl, 'BAR_2')

  bc_fixedp_n = PT.get_node_from_name_and_label(adapted_dist_zone, 'fixedp', 'BC_t')
  bc_fixedp_pl = PT.get_value(PT.get_child_from_name(bc_fixedp_n, 'PointList'))[0]
  bc_fixedp_vtx_pl = elmt_pl_to_vtx_pl(adapted_dist_zone, bc_fixedp_pl, 'BAR_2')

  ids = bc_to_rm_vtx_pl
  targets = -np.ones(bc_to_rm_vtx_pl.size, dtype=np.int32)
  vtx_distri_ini = np.array([0,n_vtx,n_vtx], dtype=np.int32) # TODO pdm_gnum
  old_to_new_vtx = merge_distributed_ids(vtx_distri_ini, ids, targets, comm, True)


  # > CHANGING TOPOLOGY !!!!!!
  update_elt_vtx_numbering(adapted_dist_zone, vtx_distri_ini, old_to_new_vtx, 'TRI_3', comm)
  n_tri = remove_elts_from_pl(adapted_dist_zone, bc_to_rm_pl, 'TRI_3', comm)
  
  tri_offset = bc_to_rm_pl.size
  apply_offset_to_elt(adapted_dist_zone, tri_offset, 'BAR_2')
  update_elt_vtx_numbering(adapted_dist_zone, vtx_distri_ini, old_to_new_vtx, 'BAR_2', comm)
  invalid_elt_pl = find_invalid_elts(adapted_dist_zone, 'BAR_2')
  n_bar = remove_elts_from_pl(adapted_dist_zone, invalid_elt_pl, 'BAR_2', comm)

  cx_n = PT.get_node_from_name(adapted_dist_zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(adapted_dist_zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(adapted_dist_zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(adapted_dist_zone)
  PT.set_value(cx_n, np.delete(cx, bc_to_rm_vtx_pl-1))
  PT.set_value(cy_n, np.delete(cy, bc_to_rm_vtx_pl-1))
  PT.set_value(cz_n, np.delete(cz, bc_to_rm_vtx_pl-1))

  PT.set_value(adapted_dist_zone, [[n_vtx-n_vtx_to_rm, n_tri, 0]])
  # PT.set_value(adapted_dist_zone, [[n_vtx, elt_range[1], 0]])


  bc_toadd = PT.get_node_from_name_and_label(adapted_dist_zone, 'Xmin', 'BC_t')


  # PT.rm_nodes_from_name(adapted_dist_zone, 'ZoneBC')
  # PT.rm_nodes_from_name(adapted_dist_zone, 'Xmax')
  # PT.rm_nodes_from_name(adapted_dist_zone, 'BAR_2.0')

  maia.io.write_tree(adapted_dist_tree, 'OUTPUT/new_mesh_wo_old_periodic_patch.cgns')


  # > Retrieve initial domain by translating periodic patch
  patch_to_move = PT.get_child_from_name(zone_bc_n, 'vol_periodic')
  patch_to_move_pl = PT.get_value(PT.get_child_from_name(bc_to_rm, 'PointList'))[0]



  sys.exit()













  if comm.Get_rank()==0:
    cgns_to_meshb(dist_tree, in_files, metric_nodes, container_names)

    # Adapt with feflo
    feflo_itp_args = f'-itp {in_file_fldb}'.split() if len(container_names)!=0 else []
    feflo_command  = ['feflo.a', '-in', str(in_files['mesh'])] + feflo_args[metric_type] + feflo_itp_args + feflo_opts.split()        
    feflo_command  = ' '.join(feflo_command) # Split + join to remove useless spaces

    mlog.info(f"Start mesh adaptation using Feflo...")
    start = time.time()
    
    subprocess.run(feflo_command, shell=True)

    end = time.time()
    mlog.info(f"Feflo mesh adaptation completed ({end-start:.2f} s)")


  # # > Recover original dist_tree
  # maia.algo.dist.redistribute_tree(dist_tree, 'uniform', comm)


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
    PT.set_value(adapted_bc, PT.get_value(input_bc))
    for node in PT.get_nodes_from_predicate(input_bc, to_copy):
      PT.add_child(adapted_bc, node)


  return adapted_dist_tree
