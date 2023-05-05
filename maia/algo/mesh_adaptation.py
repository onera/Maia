import os
import mpi4py.MPI as MPI

import maia
import maia.pytree    as PT

from maia.algo.meshb_converter import cgns_to_meshb, meshb_to_cgns

# FEFLO
feflo_path = "/home/bmaugars/tmp/feflo.a"

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
feflo_args    = { 'isotrop'  :  '-iso',
                  'mach_fld' : f'-sol {in_file_solb} -itp {in_file_fldb}',
                  'mach_hess': f'-met {in_file_solb} -itp {in_file_fldb}'
}


def mesh_adapt( dist_tree, complexity, comm, metric='mach_fld', feflo_optargs=[]):

  ''' Return a feflo adapted mesh according to a metric and a complexity.

  Adapted mesh is returned as an independant distributed tree.

  Important:
    - Input tree must be unstructured and have a element connectivity.
    - Fields in the "FSolution#Vertex#EndOfRun" node will be used by adaptation process, 
    so it must be in ``dist_tree``.

  Args:
    dist_tree     (CGNSTree)        : Distributed tree on which adaptation is done. Only U-Elements
      connectivities are managed.
    complexity    (int)             : Complexity use for the mesh adaptation process.
    comm          (MPIComm)         : MPI communicator.
    metric        (str,  optional)  : Metric used to compute the mesh adaptation.
    feflo_optargs (list, optional)  : List of feflo's optional arguments.
  Returns:
    adapted_tree (CGNSTree): Adapted mesh tree (distributed) 

  Metric choice is available through ``metric`` option with these keywords:
    - ``mach_fld`` -- Feflo's feature-based metric. Metric is computed while feflo's process.
        ``Mach`` field must be present in a "FSolution#Vertex#EndOfRun" FlowSolution_t node.
    - ``hess_mach`` -- SoNICS's feature-based metric. Metric is already computed by SoNICS, 
        ``extrap_on(sym_grad(extrap_on(#0-5`` fields must be present in a "FSolution#Vertex#EndOfRun"
        FlowSolution_t node.
    - ``iso`` -- Feflo will adapt the initial mesh into an isotrop mesh. No additional fields is required.

  Note:
    - This function has a sequential behaviour (because of the file interface with feflo).
    - Feflo mesh adaptation behaviour can be controled via feflo's arguments. You can use them through the feflo_opt argument.
    Example : ``-hgrad 2. -nordg -mesh_back mesh_back.mesh`` becomes ``["-hgrad 2.", "-nordg -mesh_back mesh_back.mesh"]``.

  '''
  assert metric in ['isotrop', 'mach_fld', 'mach_hess']

  os.system(f'mkdir -p {tmp_repo}')


  # > Gathering dist_tree on proc 0
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 

  # > CGNS to meshb conversion
  dicttag_to_bcinfo = list()
  families = list()
  if comm.Get_rank()==0:
    tree_info, dicttag_to_bcinfo, families = cgns_to_meshb(dist_tree, in_files, metric)

    avail_opt = ["p", "mesh_back", "nordg", ]

    # Adapt with feflo
    list_of_args = ['-in'  , in_files['mesh']     ,
                             feflo_args[metric],
                    '-c'   , str(complexity)      ,
                    '-cmax', str(complexity)      ] + feflo_optargs
    feflo_call = f"{feflo_path} {' '.join(list_of_args)}"
    print(f"feflo called though command line : {feflo_call}")

    os.system(feflo_call)


  # > Recover original dist_tree
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='uniform')


  # > Broadcast
  dicttag_to_bcinfo = comm.bcast(dicttag_to_bcinfo, root=0)
  families          = comm.bcast(families, root=0)

  adapted_dist_tree = meshb_to_cgns(out_files, tree_info, dicttag_to_bcinfo, families, comm, metric=='isotrop')

  return adapted_dist_tree