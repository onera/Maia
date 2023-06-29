import os
import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.utils.logging as mlog

from maia.algo.dist.meshb_converter import cgns_to_meshb, meshb_to_cgns, get_tree_info

import numpy as np 

import subprocess

from pathlib import Path


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


def unpack_metric(dist_tree, metric_paths):
  """
  Unpacks the `metric` argument from `mesh_adapt` function.
  Assert no invalid path or argument is given and that paths leads to one or six fields.
  """

  # > Get metrics nodes described by path
  if metric_paths is not None:
    if isinstance(metric_paths, str):
      base_name, zone_name, container_name, fld_name = metric_paths.split('/')
      tmp_metric_nodes = PT.get_nodes_from_names(dist_tree, [base_name, zone_name, container_name, fld_name+'*'])

      fld_names = np.array(PT.get_names(tmp_metric_nodes))
      fld_order = np.argsort(fld_names)
      fld_names = fld_names[fld_order]
      
      metric_nodes = list()
      for fld_name in fld_names:
        fld_path = '/'.join([base_name, zone_name, container_name, fld_name])
        metric_nodes.append(PT.get_node_from_path(dist_tree, fld_path))

    elif isinstance(metric_paths, list):
      assert len(metric_paths)==6, f"metric argument must be a str path or a list of 6 paths"
      metric_nodes = list()
      for path in metric_paths:
        base_name, zone_name, container_name, fld_name = path.split('/')
        metric_nodes.append(PT.get_node_from_path(dist_tree, path))

    else:
      raise ValueError(f"Incorrect metric type (expected list or str).")


    # > Assert that metric is one or six fields
    if len(metric_nodes)==0:
      raise ValueError(f"Metric path \"{metric_paths}\" is invalid.")
    if len(metric_nodes)!=1 and len(metric_nodes)!=6:
      raise ValueError(f"Metric path \"{metric_paths}\" leads to {len(metric_nodes)} nodes (1 or 6 expected).")

  else:
    metric_nodes = list()

  print(f"TODO : \n \
     - pk get_nodes_from_names et get_node_from_path ont pas le meme comportement ?\n\
     - get_nodes_from_names bien utilisée ?\n\
     ")

  return metric_nodes



def mesh_adaptation(dist_tree, comm, metric=None, container_names=[], feflo_options=[]):
  ''' Return a feflo adapted mesh according to a metric and options.

  Adapted mesh is returned as an independant distributed tree.

  Important:
    - Input tree must be unstructured and have a element connectivity.

  Args:
    dist_tree      (CGNSTree)    : Distributed tree on which adaptation is done. Only U-Elements
      connectivities are managed.
    comm           (MPIComm)     : MPI communicator.
    metric         (str or list) : Path(s) to metric fields.
    container_names(list)        : Container names that must been projected on adapted mesh (Vertex Center).
    feflo_options (list)        : List of feflo's optional arguments.
  Returns:
    adapted_tree (CGNSTree): Adapted mesh tree (distribute).

  Metric choice is available through type of ``metric`` argument.
  If it's str :
    - if path leads to 1 field  in CGNSTree -- Feflo's feature-based metric.
      Metric is computed with this field while feflo's process.
    - if path leads to 6 fields in CGNSTree -- User's  feature-based metric.
      Metric is already computed, and will be used by feflo (must be stored 
      with suffix (``XX``,``XY``,``XZ``,``YY``,``YZ``,``ZZ``), order fixed by maia).
  If it's list -- User's feature-based metric -- Metric is already computed,
  and will be used by feflo with order described by user in the list.
    - if no paths are given, feflo will adapt the initial mesh into an isotrop mesh.


  Note:
    - Each BC from dist_tree must be FamilySpecified.
    - This function interface is parallel, but because feflo is sequential, dist_tree is reduced
    to one proc to perform mesh adaptation. Beware about memory issues !
    - Feflo mesh adaptation behaviour can be controled via feflo's arguments. You can use them through the feflo_options argument.
    Example : ``-hgrad 2. -nordg -mesh_back mesh_back.mesh`` becomes ``["-hgrad", "2.", "-nordg", "-mesh_back", "mesh_back.mesh"]``.
    - Note that feflo's metric order is ``XX``,``XY``,``YY``,``XZ``,``YZ``,``ZZ``,
    so beware about list order while passing a `metric` list argument.
    - Feflo's binary must be in user's profile (Example : `export PATH='path_to_feflo_dir':$PATH)

  '''

  Path(tmp_repo).mkdir(exist_ok=True)

  # > Get metric nodes
  metric_nodes = unpack_metric(dist_tree, metric)
  metric_names = PT.get_names(metric_nodes)

  n_metric_path = len(metric_nodes)
  if   n_metric_path==0: metric_type = 'isotrop'
  elif n_metric_path==1: metric_type = 'from_fld'
  elif n_metric_path==6: metric_type = 'from_hess'


  # > Get tree structure and names
  tree_info = get_tree_info(dist_tree, container_names)


  # > Gathering dist_tree on proc 0
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 


  # > CGNS to meshb conversion
  dicttag_to_bcinfo = list()

  if comm.Get_rank()==0:
    cgns_to_meshb(dist_tree, in_files, metric_nodes, container_names)

    # Adapt with feflo
    feflo_call_list = ["feflo.a"]             \
                    + ['-in', in_files['mesh']]\
                    + feflo_args[metric_type]  \
                    + feflo_options                

    mlog.info(f"Feflo mesh adaptation...")
    start = time.time()
    
    subprocess.run(feflo_call_list, shell=True)

    end = time.time()
    mlog.info(f"Feflo mesh adaptation completed ({end-start:.2f} s) --")


  # > Recover original dist_tree
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='uniform')


  # > Get adapted dist_tree
  adapted_dist_tree = meshb_to_cgns(out_files, tree_info, comm)


  return adapted_dist_tree