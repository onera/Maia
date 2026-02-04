# Do not modify this section
# -----------------------------------------------------------------------------
def get_solver_tree():
    tree  = maia.io.file_to_dist_tree('airplane.cgns', comm)
    ptree = maia.factory.partition_dist_tree(tree, comm,
            preserve_orientation=True, data_transfer='FIELDS')
    return ptree

def plot_1d_profile(x, y, comm):
    import numpy as np
    import plotly.graph_objects as go

    x = comm.gather(x, root=0)
    y = comm.gather(y, root=0)
    
    if comm.Get_rank() != 0:
        return

    # > Concatenate arrays
    x = np.concatenate(x)
    y = np.concatenate(y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Wing',
        )
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text='X position',
            ),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
            ),
        yaxis=dict(
            title=dict(
                text='Pressure',
            ),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        ),
        font=dict(
            family='Courier New, monospace',
            size=14,
        ),
        plot_bgcolor='white',
        showlegend=False,
    )

    fig.write_image('1dprofile.png')
# -----------------------------------------------------------------------------

# Warmup
from mpi4py import MPI
import numpy as np

import maia
import maia.pytree as PT

comm = MPI.COMM_WORLD

tree = get_solver_tree()