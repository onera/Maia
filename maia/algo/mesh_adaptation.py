import os
import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.utils.logging as mlog

from maia.algo.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info

import subprocess

from pathlib import Path


# FEFLO
feflo_path = "feflo" # TODO : Must be an alias in user profile
feflo_path = "/home/bmaugars/tmp/feflo.a" # Must be an alias in user profile

# TMP directory
tmp_repo   = 'TMP_adapt_repo/'

# INPUT files
in_file_meshb = tmp_repo + 'mesh.mesh'
in_file_solb  = tmp_repo + 'metric.sol'
in_file_fldb  = tmp_repo + 'field.sol'
in_files = {'mesh': in_file_meshb,
            'sol' : in_file_solb ,
            'fld' : in_file_fldb }

mesh_back_file = tmp_repo + 'mesh_back.mesh'

# OUTPUT files
out_file_meshb = tmp_repo + 'mesh.o.mesh'
out_file_solb  = tmp_repo + 'mesh.o.sol'
out_file_fldb  = tmp_repo + 'field.itp.sol'
out_files = {'mesh': out_file_meshb,
             'sol' : out_file_solb ,
             'fld' : out_file_fldb }

# Feflo files arguments
feflo_args    = { 'isotrop'  : f'-iso                -itp {in_file_fldb}'.split(),
                  'from_fld' : f'-sol {in_file_solb} -itp {in_file_fldb}'.split(),
                  'from_hess': f'-met {in_file_solb} -itp {in_file_fldb}'.split()
}


# def mesh_adapt( dist_tree, complexity, comm, metric='from_fld', feflo_opt=[]):
def mesh_adapt(dist_tree, comm, metric=[], container_names=None, feflo_opt=[]):
  ''' Return a feflo adapted mesh according to a metric and a complexity.

  Adapted mesh is returned as an independant distributed tree.

  Important:
    - Input tree must be unstructured and have a element connectivity.
    - Fields in the "FSolution#Vertex#EndOfRun" node will be used by adaptation process, 
    so it must be in ``dist_tree``.

  Args:
    dist_tree     (CGNSTree)        : Distributed tree on which adaptation is done. Only U-Elements
      connectivities are managed.
    comm          (MPIComm)         : MPI communicator.
    metric         (list)           : Paths to metric fields.
    container_names(list)           : Container names that must been projected on adapted mesh (Vertex Center)
    feflo_opt     (list, optional)  : List of feflo's optional arguments.
  Returns:
    adapted_tree (CGNSTree): Adapted mesh tree (distributed) 

  Metric choice is available through number of ``metric`` path given. Paths is used as a preffix
    - if paths leads to 1 field  in CGNSTree -- Feflo's feature-based metric. Metric is computed while feflo's process on this field.
    - if paths leads to 6 fields in CGNSTree -- User's feature-based metric. Metric is already computed, and will be used by feflo.
    (Must be stored with suffix (``XX``,``XY``,``XZ``,``YY``,``YZ``,``ZZ``,))
    - if no paths are given, feflo will adapt the initial mesh into an isotrop mesh.

  Note:
    - This function has a sequential behaviour (because of the file interface with feflo).
    - Feflo mesh adaptation behaviour can be controled via feflo's arguments. You can use them through the feflo_opt argument.
    Example : ``-hgrad 2. -nordg -mesh_back mesh_back.mesh`` becomes ``["-hgrad", "2.", "-nordg", "-mesh_back", "mesh_back.mesh"]``.

  '''

  Path(tmp_repo).mkdir(exist_ok=True)
  # Path.cwd()/Path(tmp_repo).mkdir(exist_ok=True)

  metric_nodes = list()
  for path in metric:
    base_name, zone_name, container_name, fld_name = path.split('/')
    metric_nodes += PT.get_nodes_from_names(dist_tree, [base_name, zone_name, container_name, fld_name+'*'])
  print(f"len(metric_nodes) = {len(metric_nodes)}")
  n_metric_fld = len(metric_nodes)
  assert n_metric_fld in [0,1,6]
  if   n_metric_fld==0 : metric_name = 'isotrop'
  elif n_metric_fld==1 : metric_name = 'from_fld'
  elif n_metric_fld==6 : metric_name = 'from_hess'

  # > Get tree structure and names
  tree_info = get_tree_info(dist_tree)


  # > Gathering dist_tree on proc 0
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 


  # > CGNS to meshb conversion
  dicttag_to_bcinfo = list()

  if comm.Get_rank()==0:
    cgns_to_meshb(dist_tree, in_files, metric_nodes, container_names)

    # Adapt with feflo
    print(f"feflo_call_list = {['-in', in_files['mesh']]+ feflo_args[metric_name]}")
    print(f"feflo_opt       = {feflo_opt}")
    feflo_call_list = [feflo_path]             \
                    + ['-in', in_files['mesh']]\
                    + feflo_args[metric_name]  \
                    + feflo_opt                
    print(f"feflo_call_list = {feflo_call_list}")

    mlog.info(f"Feflo mesh adaptation...")
    start = time.time()
    
    subprocess.run(feflo_call_list)

    end = time.time()
    mlog.info(f"Feflo mesh adaptation completed ({end-start:.2f} s) --")


  # > Recover original dist_tree
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='uniform')


  dicttag_to_bcinfo = tree_info["dicttag_to_bcinfo"]
  print(f"dicttag_to_bcinfo = {dicttag_to_bcinfo}")
  for loc, tag_to_bcinfo in dicttag_to_bcinfo.items():
      print(f"\nLOC = {loc}")
      for tag, bcinfo in tag_to_bcinfo.items():
          print(f" - TAG = {tag} -> BC = {bcinfo['BC']:10} ; FAMILY = {bcinfo['Family']}")
          # PT.print_tree(bcinfo)

  adapted_dist_tree = meshb_to_cgns(out_files, tree_info, comm)

  return adapted_dist_tree