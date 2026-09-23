import numpy as np
import plotly.graph_objects as pgo
import plotly.io as pio
from scotty.geometry_v4 import MagneticField_Cartesian, MagneticField_Cylindrical
from scotty.typing import FloatArray
from typing import Union, Tuple

def plot_3d_beam(
    field: Union[MagneticField_Cartesian, MagneticField_Cylindrical],
    q_X: FloatArray,
    q_Y: FloatArray,
    q_Z: FloatArray,
    widths_N: FloatArray,
    widths_B: FloatArray,
    n_hat: FloatArray,
    b_hat: FloatArray,
    theta_resolution: int = 60,
    show_in_browser: bool = True,
    xlims: Tuple[float, float] = (0.5, 2.5),
    ylims: Tuple[float, float] = (-0.5, 0.5),
    zlims: Tuple[float, float] = (-1, 1),
):
    """
    Vectorised 3D beam visualization using Plotly Mesh3d
    
    q_X, q_Y, q_Z : (N,) FloatArray
        Central ray coordinates
    widths_N, widths_B : (N,) FloatArray
        Beam radii along N(ormal) and B(inormal) directions
    n_hat, b_hat : (N,3) FloatArray
        N(ormal) and B(inormal) unit vectors
    theta_resolution : int
        Angular resolution in the beam cross-section
    show_in_browser : bool
        If True, then the graph is displayed in the browser
    """

    # Vectorise the beam silhouette (beam_surf)
    q_XX = q_X[:, None] # shape (N,1)
    q_YY = q_Y[:, None] # shape (N,1)
    q_ZZ = q_Z[:, None] # shape (N,1)
    w_NN = widths_N[:, None] # shape (N,1)
    w_BB = widths_B[:, None] # shape (N,1)

    n_T = n_hat.T # shape (3,N)
    b_T = b_hat.T # shape (3,N)

    theta = np.linspace(0, 2*np.pi, theta_resolution)
    sin_theta = np.sin(theta)[None, :] # shape (1,T)
    cos_theta = np.cos(theta)[None, :] # shape (1,T)

    X_beam_surf = (q_XX + (w_NN * cos_theta * n_T[0][:, None]) + (w_BB * sin_theta * b_T[0][:, None])).ravel()
    Y_beam_surf = (q_YY + (w_NN * cos_theta * n_T[1][:, None]) + (w_BB * sin_theta * b_T[1][:, None])).ravel()
    Z_beam_surf = (q_ZZ + (w_NN * cos_theta * n_T[2][:, None]) + (w_BB * sin_theta * b_T[2][:, None])).ravel()

    # Get triangular mesh indices for the corresponding points
    # of the beam silhouette, where the vertices are connected like
    # triangle set 1: (i, j) -> (i, j+1) -> (i+1, j+1)
    # triangle set 2: (i, j) -> (i+1, j+1) -> (i+1, j)
    # and stack them all together
    ii = np.arange(len(q_X))
    jj = np.arange(theta_resolution)
    I, J = np.meshgrid(ii[:-1], jj[:-1], indexing="ij")
    I, J = I.ravel(), J.ravel()

    def tri_map(i, j): return i*theta_resolution + j
    t_i = np.concatenate([tri_map(I, J),     tri_map(I, J)])
    t_j = np.concatenate([tri_map(I, J+1),   tri_map(I+1, J+1)])
    t_k = np.concatenate([tri_map(I+1, J+1), tri_map(I+1, J)])

    # Some plotly stuff
    if show_in_browser: pio.renderers.default = "browser"
    fig = pgo.Figure()

    # Plot flux surfaces
    # remember to set alpha=0.5 or sth
    # <--- TO DO

    # Plot central ray
    fig.add_trace(pgo.Scatter3d(
        x=q_X,
        y=q_Y,
        z=q_Z,
        mode="lines",
        line=dict(color="black", width=6),
        name="Central ray (plasma)"
    ))

    # Plot beam surface
    fig.add_trace(pgo.Mesh3d(
        x=X_beam_surf,
        y=Y_beam_surf,
        z=Z_beam_surf,
        i=t_i,
        j=t_j,
        k=t_k,
        color="black",
        opacity=0.25,
        name="1/e Beam (plasma)"
    ))

    fig.update_layout(
        title="Testing",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            xaxis=dict(range=xlims),
            yaxis=dict(range=ylims),
            zaxis=dict(range=zlims),
        ),
        width=900,
        height=700
    )

    fig.show()