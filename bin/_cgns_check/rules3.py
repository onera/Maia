from mpi4py import MPI
import numpy as np
import sys
import inspect

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.typing import List, CGNSTree, MPIComm # Strangly import * brings NamedTuple as a function



OK = ''



# Errors code 300-399


def eso_values(nodes:List[CGNSTree], comm:MPIComm) -> str: 
    """E301 - ElementStartOffset values
    
    ElementStartOffset arrays should have the following properties:
        - first value is 0
        - array is stricly increasing (eso[i] < eso[i+1])
        - last value is the size of related ElementConnectivity array
    
    Erroneous tree example:

    NGonElements Elements_t I4 [22 0]:
    ├───ElementRange IndexRange_t I4 [13 16]
    ├───ElementStartOffset DataArray_t I4 [0 4 \033[91m14\033[m 12 16]
    └───ElementConnectivity DataArray_t I4 (16,)
    """
    # Careful : we need an additional rule that report missing ESO on NGON/NFACE/MIXED
    last = nodes[-1]
    if PT.get_name(last) == 'ElementStartOffset' and PT.get_label(nodes[-2]) == 'Elements_t':
        eso = PT.get_np_value(last)
        if not (st:=comm.bcast(eso[0], root=0)) == 0:
            return f"ESO array should start at 0, but starting value is {st}"
            
        mask = ~(eso[:-1] < eso[1:])
        lsum = mask.sum()
        if (gsum:=comm.allreduce(lsum)) > 0:
            distri = MT.distribution_value(nodes[-2], 'Element')
            lval = np.where(mask)[0][0] + distri[0] if lsum > 0 else distri[2]+1
            gval = comm.allreduce(lval, MPI.MIN)
            return f"ESO array is not strictly increasing (ESO[i] < ESO[i+1]) : {gsum} indices are wrong, first one beeing {gval}"

    return OK



_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)

DNODE_RULES = {func[1].__doc__[:4] : func[1] for func in _funcs}
assert len(DNODE_RULES) == len(_funcs)