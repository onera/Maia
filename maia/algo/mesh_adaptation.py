import os
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT

# FEFLO
feflo_path = "/stck/jvanhare/wkdir/spiro/bin/feflo.a"

# PATHS
tmp_repo   = 'TMP_adapt_repo'
file_meshb = tmp_repo + 'initial_mesh.mesh'
file_solb  = tmp_repo + 'initial_mesh.sol'
file_fldb  = tmp_repo + 'initial_mesh.fld'

# Feflo files arguments
feflo_args    = { 'isotrop'  :  '-iso'
                  'mach_fld' : f'-sol {file_solb} -itp {file_fldb}'
                  'mach_hess': f'-met {file_solb} -itp {file_fldb}'
}


def mesh_adaptation(dist_tree, complexity, comm,
                    tool='feflo', criterion='mach_fld', feflo_optargs='-p 4 -hgrad=1.'):

  assert tool      in ['feflo']
  assert criterion in ['isotrop', 'mach_fld', 'mach_hess']

  os.system(f'mkdir {tmp_repo}')

  # Gathering dist_tree on proc 0
  if rank==0 : print('[MAIA] gather dist_tree on proc0')
  Mad.redistribute_tree(dist_tree, comm, policy='gather') # Modifie le dist_tree 

  # CGNS to meshb conversion
  if rank==0 : print('[MAIA] cgns to meshb conversion')

  dicttag_to_bcinfo = list()
  families = list()
  if rank==0:
    print('[MAIA] CGNS to meshb')
    dicttag_to_bcinfo, families = cgns_to_meshb(dist_tree, tmp_repo, output_name)

    # Adapt with feflo
    print('[MAIA] feflotation')
    os.system(f"{feflo_path}  -in   {file_meshb} \
                                    {feflo_options[criterion]}\
                                    {feflo_optargs} \
                              -c    {complexity} \
                              -cmax {complexity}")