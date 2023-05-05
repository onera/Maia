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
in_file_solb  = tmp_repo + 'criterion.sol'
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



def mesh_adapt( dist_tree, complexity, comm,
                tool='feflo', criterion='mach_fld', feflo_optargs='-p 4 -hgrad=1.'.split(" "),
                keep_mesh_back=False):

  '''
  * Gerer le maillage back (avec le patch paradigm) :
      - garder le meshb du premier passage -> maillage_back.mesh
      - ajouter  "-nordg -back maillage_back.mesh" aux itérations suivantes
      /home/bmaugars/dev/dev-SoNICS/SONICE-2023/debug_spalart/sonics/test/cases/cgns_to_meshb.py
  '''
  assert tool      in ['feflo']
  assert criterion in ['isotrop', 'mach_fld', 'mach_hess']

  # os.system(f'rm -rf {tmp_repo}')
  os.system(f'mkdir -p {tmp_repo}')

  # Gathering dist_tree on proc 0
  maia.algo.dist.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 

  # CGNS to meshb conversion
  dicttag_to_bcinfo = list()
  families = list()
  if comm.Get_rank()==0:
    tree_info, dicttag_to_bcinfo, families = cgns_to_meshb(dist_tree, in_files, criterion)

    if keep_mesh_back : os.system(f"cp {in_files['mesh']} {mesh_back_file}")

    # Adapt with feflo
    list_of_args = ['-in'  , in_files['mesh']     ,
                             feflo_args[criterion],
                    '-c'   , str(complexity)      ,
                    '-cmax', str(complexity)      ] + feflo_optargs
    print(' '.join(list_of_args))

    # os.system(f"{feflo_path}  -in   {in_files['mesh']} \
    #                                 {feflo_args[criterion]}\
    #                           -c    {complexity} \
    #                           -cmax {complexity} \
    #                                 {feflo_optargs}")
    os.system(f"{feflo_path} {' '.join(list_of_args)}")


  # Broadcast
  dicttag_to_bcinfo = comm.bcast(dicttag_to_bcinfo, root=0)
  families          = comm.bcast(families, root=0)

  dist_tree = meshb_to_cgns(out_files, tree_info, dicttag_to_bcinfo, families, comm, criterion=='isotrop')

  return dist_tree