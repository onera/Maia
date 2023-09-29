import time
import mpi4py.MPI as MPI

import maia
from   maia.algo.part.extraction_utils   import local_pl_offset, LOC_TO_DIM
import maia.pytree        as PT
import maia.utils.logging as mlog
from   maia.utils         import np_utils
from maia.io.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info


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
  PT.print_tree(dist_tree)

  dist_zone = PT.get_nodes_from_label(dist_tree, 'Zone_t')
  assert len(dist_zone)==1
  dist_zone = dist_zone[0]

  gc_nodes = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  assert len(gc_nodes)==2
  gc_names = [PT.get_name(n) for n in gc_nodes]
  gc_n  = gc_nodes[0]
  print(f'GC used: {PT.get_name(gc_n)}')
  gc_pl  = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))[0]
  gc_pld = PT.get_value(PT.get_child_from_name(gc_n, 'PointListDonor'))[0]
  assert PT.Subset.GridLocation(gc_n)=='EdgeCenter' # 2d
  is_bar_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)=='BAR_2'
  bar_nodes = PT.get_nodes_from_predicate(dist_tree, is_bar_elt)
  print(len(bar_nodes))
  bar_con   = [PT.get_value(PT.get_child_from_name(bar_n, 'ElementConnectivity')) for bar_n in bar_nodes]
  print(f'bar_con = {[len(bar_c) for bar_c in bar_con]}')
  bar_con   = np_utils.concatenate_np_arrays(bar_con)[1]
  print(f'bar_con = {bar_con}')
  gc_pl  = gc_pl  - local_pl_offset(dist_zone, LOC_TO_DIM['EdgeCenter']) -1
  gc_pld = gc_pld - local_pl_offset(dist_zone, LOC_TO_DIM['EdgeCenter']) -1
  con_gc_pl  = np_utils.interweave_arrays([2*gc_pl , 2*gc_pl +1])
  con_gc_pld = np_utils.interweave_arrays([2*gc_pld, 2*gc_pld+1])

  gc_vtx_pl  = bar_con[con_gc_pl ]
  gc_vtx_pld = bar_con[con_gc_pld]
  gc_vtx_pl  = np.unique(gc_vtx_pl )
  gc_vtx_pld = np.unique(gc_vtx_pld)
  print(f'gc_vtx_pl  = {gc_vtx_pl }')
  print(f'gc_vtx_pld = {gc_vtx_pld}')
  # bc_vtx = 

  # > Which cells are connected to donor_vtx ?
  n_vtx = PT.Zone.n_vtx(dist_zone)
  n_tri = PT.Zone.n_cell(dist_zone)
  print(f'n_tri = {n_tri}')
  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and\
                         PT.Element.CGNSName(n)=='TRI_3'
  tri_nodes = PT.get_nodes_from_predicate(dist_tree, is_tri_elt)
  tri_con   = [PT.get_value(PT.get_child_from_name(tri_n, 'ElementConnectivity')) for tri_n in tri_nodes]
  tri_con   = np_utils.concatenate_np_arrays(tri_con)[1]
  print(f'tri_con = {tri_con}')
  
  tag_vtx   = np.isin(tri_con, gc_vtx_pld) # True where vtx is 
  tag_tri   = np.logical_or.reduceat(tag_vtx, np.arange(0,n_tri*3,3)) # True when has vtx 
  gc_tri_pl = np.where(tag_tri)[0] # Which cells has vtx
  print(f'gc_tri_pl = {gc_tri_pl}')

  n_tri_period = gc_tri_pl.size
  print(f'n_tri_period = {n_tri_period}')
  ptri_con = -np.ones(n_tri_period*3, dtype=np.int32)
  ptri_con[0::3] = tri_con[3*gc_tri_pl+0]
  ptri_con[1::3] = tri_con[3*gc_tri_pl+1]
  ptri_con[2::3] = tri_con[3*gc_tri_pl+2]
  print(f'ptri_con   = {ptri_con}')
  tag_pvtx  = np.isin(ptri_con, gc_vtx_pld) # True where vtx is 
  print(f'tag_pvtx = {tag_pvtx}')
  print(f'not tag_pvtx = {np.invert(tag_pvtx)}')
  gc_pvtx_pl1 = np.where(          tag_pvtx )[0] # Which vtx is in gc
  gc_pvtx_pl2 = np.where(np.invert(tag_pvtx))[0] # Which vtx is not in gc
  plast_vtx = np.unique(ptri_con[gc_pvtx_pl2])
  print(f'gc_vtx_pld = {gc_vtx_pld}')
  print(f'gc_pvtx_pl1   = {gc_pvtx_pl1}')
  print(f'gc_pvtx_pl2   = {gc_pvtx_pl2}')
  print(f'\n\nplast_vtx = {plast_vtx}')
  n_plast_vtx = plast_vtx.size
  vtx_transfo = {k:v   for k, v in zip(gc_vtx_pld, gc_vtx_pl)}
  new_num_vtx = {k:v+1 for k, v in zip(plast_vtx, np.arange(n_vtx, n_vtx+n_plast_vtx))}
  print(f'n_plast_vtx = {n_plast_vtx}')
  print(f'ptri_con = {ptri_con}')

  print(f'vtx_transfo = {vtx_transfo}')
  print(f'new_num_vtx = {new_num_vtx}')
  print(f'ptri_con[gc_pvtx_pl1] = {ptri_con[gc_pvtx_pl1]}')
  ptri_con[gc_pvtx_pl1] = [vtx_transfo[i_vtx] for i_vtx in ptri_con[gc_pvtx_pl1]]
  ptri_con[gc_pvtx_pl2] = [new_num_vtx[i_vtx] for i_vtx in ptri_con[gc_pvtx_pl2]]
  print(f'ptri_con = {ptri_con}')

  cx_n = PT.get_node_from_name(dist_zone, 'CoordinateX')
  cy_n = PT.get_node_from_name(dist_zone, 'CoordinateY')
  cz_n = PT.get_node_from_name(dist_zone, 'CoordinateZ')
  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  pcx = -np.ones(n_plast_vtx, dtype=np.float64)
  pcy = -np.ones(n_plast_vtx, dtype=np.float64)
  pcz = -np.ones(n_plast_vtx, dtype=np.float64)
  pcx = cx[plast_vtx-1] -1.
  pcy = cy[plast_vtx-1]
  pcz = cz[plast_vtx-1]
  PT.set_value(cx_n, np.concatenate([cx, pcx]))
  PT.set_value(cy_n, np.concatenate([cy, pcy]))
  PT.set_value(cz_n, np.concatenate([cz, pcz]))
  PT.print_tree(cx_n, verbose=True)
  PT.print_tree(cy_n, verbose=True)
  PT.print_tree(cz_n, verbose=True)

  tri_n = PT.get_node_from_name(dist_tree, 'TRI_3.0')
  PT.print_tree(tri_n)
  print(f'tri_con  = {tri_con}')
  print(f'ptri_con = {ptri_con}')
  tri_con = np.concatenate([tri_con, ptri_con])
  tri_con_n = PT.get_child_from_name(tri_n, 'ElementConnectivity')
  PT.set_value(tri_con_n, tri_con)
  elt_range_n = PT.get_child_from_name(tri_n, 'ElementRange')
  elt_range = PT.get_value(elt_range_n)
  n_tri_new = elt_range[1]+n_tri_period
  elt_range[1] = n_tri_new
  PT.set_value(elt_range_n, elt_range)

  PT.set_value(dist_zone, [[n_vtx+n_plast_vtx, elt_range[1], 0]])


  tri_n = PT.get_node_from_name(dist_tree, 'TRI_3.0')
  tri_elt_range_n = PT.get_child_from_name(tri_n, 'ElementRange')
  tri_elt_range = PT.get_value(tri_elt_range_n)

  bar_n = PT.get_node_from_name(dist_tree, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range = PT.get_value(bar_elt_range_n)
  bar_offset = tri_elt_range[1]-bar_elt_range[0]+1
  bar_elt_range[0] = bar_elt_range[0]+bar_offset
  bar_elt_range[1] = bar_elt_range[1]+bar_offset
  PT.set_value(bar_elt_range_n, bar_elt_range)
  # bar_elt_conn_n = PT.get_child_from_name(bar_n, 'ElementConnectivity')
  # bar_elt_conn   = PT.get_value(bar_elt_conn_n)
  # bar_elt_conn   = bar_elt_conn+bar_offset
  # PT.set_value(bar_elt_conn_n, bar_elt_conn)

  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and\
                         PT.Subset.GridLocation(n)=='EdgeCenter'
  for edge_bc_n in PT.get_nodes_from_predicate(dist_tree, is_edge_bc):
    pl_n = PT.get_child_from_name(edge_bc_n, 'PointList')
    pl = PT.get_value(pl_n)
    PT.set_value(pl_n, pl+bar_offset)

  zone_bc_n = PT.get_node_from_label(dist_tree, 'ZoneBC_t')
  for gc_n in gc_nodes:
    gc_name = PT.get_name(gc_n)
    gc_pl   = PT.get_value(PT.get_child_from_name(gc_n, 'PointList'))
    PT.new_BC(name=gc_name,
              type='BCWall',
              point_list=gc_pl+bar_offset,
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)


  # > Ajout des nouvelles BCs
  pbar_conn = -np.ones((n_plast_vtx-1)*2, dtype=np.int32)
  pbar_conn[0::2] = np.array(list(new_num_vtx.values()), dtype=np.int32)[0:-1]
  pbar_conn[1::2] = np.array(list(new_num_vtx.values()), dtype=np.int32)[1:]
  print(f'\n\n pbar_conn = {pbar_conn}')
  bar_n = PT.get_node_from_name(dist_tree, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range   = PT.get_value(bar_elt_range_n)
  n_bar = bar_elt_range[1]-bar_elt_range[0]+1
  bar_elt_range[1]= bar_elt_range[1]+(n_plast_vtx-1)
  PT.set_value(bar_elt_range_n, bar_elt_range)
  bar_elt_conn_n = PT.get_child_from_name(bar_n, 'ElementConnectivity')
  bar_elt_conn   = PT.get_value(bar_elt_conn_n)
  bar_elt_conn   = np.concatenate([bar_elt_conn, pbar_conn])
  PT.set_value(bar_elt_conn_n, bar_elt_conn)

  pbar_conn = -np.ones((n_plast_vtx-1)*2, dtype=np.int32)
  pbar_conn[0::2] = np.array(list(new_num_vtx.keys()), dtype=np.int32)[0:-1]
  pbar_conn[1::2] = np.array(list(new_num_vtx.keys()), dtype=np.int32)[1:]
  print(f'\n\n pbar_conn = {pbar_conn}')
  bar_n = PT.get_node_from_name(dist_tree, 'BAR_2.0')
  bar_elt_range_n = PT.get_child_from_name(bar_n, 'ElementRange')
  bar_elt_range   = PT.get_value(bar_elt_range_n)
  bar_elt_range[1]= bar_elt_range[1]+(n_plast_vtx-1)
  PT.set_value(bar_elt_range_n, bar_elt_range)
  bar_elt_conn_n = PT.get_child_from_name(bar_n, 'ElementConnectivity')
  bar_elt_conn   = PT.get_value(bar_elt_conn_n)
  bar_elt_conn   = np.concatenate([bar_elt_conn, pbar_conn])
  PT.set_value(bar_elt_conn_n, bar_elt_conn)

  PT.print_tree(bar_n)
  PT.new_BC(name='fixed',
              type='BCWall',
              point_list=np.arange(bar_elt_range[0]+n_bar, bar_elt_range[0]+n_bar+(n_plast_vtx-1)*2, dtype=np.int32).reshape((1,-1), order='F'),
              loc='EdgeCenter',
              family='BCS',
              parent=zone_bc_n)


  PT.rm_nodes_from_name(dist_tree, 'ZoneGridConnectivity')
  maia.io.write_tree(dist_tree, 'OUTPUT/square_extended.cgns')

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

  '''

  # sys.exit()

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
