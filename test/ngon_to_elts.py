import Converter.PyTree as C

import maia.transform.transform2 as maia

path = "/scratchm/bberthou/cases/mesh_database/M6_ACI_cassiopee/"
t = C.convertFile2PyTree(path+'elsA_withBC.hdf', 'bin_hdf')
maia.convert_from_ngon_to_simple_connectivities(t)
C.convertPyTree2File(t,'elsA_out.hdf', 'bin_hdf')
