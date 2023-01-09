import os
import mpi4py.MPI as MPI

import maia
import maia.pytree    as PT
import maia.algo.dist as Mad

from maia.algo.meshb_converter import cgns_to_meshb
from maia.algo.meshb_converter import meshb_to_cgns

# FEFLO
feflo_path = "/stck/jvanhare/wkdir/spiro/bin/feflo.a"


# TMP directory
tmp_repo   = 'TMP_adapt_repo/'

# INPUT files
in_file_meshb = tmp_repo + 'mesh.mesh'
in_file_solb  = tmp_repo + 'criterion.sol'
in_file_fldb  = tmp_repo + 'field.sol'
in_files = {'mesh': in_file_meshb,
            'sol' : in_file_solb ,
            'fld' : in_file_fldb }

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



def mesh_adapt(dist_tree, complexity, comm,
               tool='feflo', criterion='mach_fld', feflo_optargs='-p 4 -hgrad=1.'):

  assert tool      in ['feflo']
  assert criterion in ['isotrop', 'mach_fld', 'mach_hess']

  os.system(f'rm -rf {tmp_repo}')
  os.system(f'mkdir  {tmp_repo}')

  # Gathering dist_tree on proc 0
  Mad.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 

  # CGNS to meshb conversion
  dicttag_to_bcinfo = list()
  families = list()
  if comm.Get_rank()==0:
    dicttag_to_bcinfo, families = cgns_to_meshb(dist_tree, in_files, criterion)

    # Adapt with feflo
    os.system(f"{feflo_path}  -in   {in_files['mesh']} \
                                    {feflo_args[criterion]}\
                                    {feflo_optargs} \
                              -c    {complexity} \
                              -cmax {complexity}")


  # Broadcast
  dicttag_to_bcinfo = comm.bcast(dicttag_to_bcinfo, root=0)
  families          = comm.bcast(families, root=0)

  dist_tree = meshb_to_cgns(out_files, dicttag_to_bcinfo, families, comm, criterion=='isotrop')

  return dist_tree