import logging
import numpy as np
from scipy.integrate import solve_ivp
from scotty.geometry_v4 import MagneticField_Cartesian
from scotty.hamiltonian_v4 import Hamiltonian, hessians
from scotty.logger_v4 import timer
from scotty.typing import FloatArray, ComplexFloatArray
from typing import Tuple

log = logging.getLogger(__name__)

UPPER_TRI_INDICES = np.triu_indices(3)

def pack_beam_parameters(q: FloatArray, K: FloatArray, Psi: ComplexFloatArray) -> FloatArray:
    
    """Packs `q` (shape `(3, N)`), `K` (shape `(3, N)`), and `Psi`
    (shape `(6, N)`) into an array of shape `(18, N)`. Note that `Psi` is 
    decomposed into its real and imaginary components, which each have
    shape `(6, N)`"""
    
    log.trace(f"Packing ray and beam parameters")
    N = q.shape[1] if q.ndim == 2 else 1
    upper_Psi = Psi[UPPER_TRI_INDICES]

    q_K_RePsi_ImPsi_arr = np.empty((18, N), dtype=np.float64)
    q_K_RePsi_ImPsi_arr[0:3] = q
    q_K_RePsi_ImPsi_arr[3:6] = K
    q_K_RePsi_ImPsi_arr[6:12]  = upper_Psi.real
    q_K_RePsi_ImPsi_arr[12:18] = upper_Psi.imag

    return q_K_RePsi_ImPsi_arr



def unpack_beam_parameters(q_K_RePsi_ImPsi_arr: FloatArray) -> Tuple[FloatArray, FloatArray, ComplexFloatArray]:

    """Unpacks `q_K_RePsi_ImPsi_arr` of shape `(18, N)` back into
    `q` (shape `(3, N)`), `K` (shape `(3, N)`), and `Psi` (shape `(6, N)`).
    Note that `Psi` is recomposed into its real and imaginary components,
    which each have shape `(6, N)`"""

    log.trace(f"Unpacking ray and beam parameters")
    q = q_K_RePsi_ImPsi_arr[0:3]
    K = q_K_RePsi_ImPsi_arr[3:6]
    Psi_upper = q_K_RePsi_ImPsi_arr[6:12] + 1j * q_K_RePsi_ImPsi_arr[12:18]

    N = q_K_RePsi_ImPsi_arr.shape[1] if q_K_RePsi_ImPsi_arr.ndim == 2 else 1
    i, j = UPPER_TRI_INDICES
    Psi = np.empty((N, 3, 3), dtype=np.complex128)
    Psi[:, i, j] = Psi_upper.T
    Psi[:, j, i] = Psi[:, i, j]

    return q, K, Psi



def d_beam_parameters_d_tau(beam_parameters: FloatArray, hamiltonian: Hamiltonian) -> FloatArray:
    
    """
    something
    """

    q, K, Psi = unpack_beam_parameters(beam_parameters)

    dH = hamiltonian.derivatives(q, K, second_order=True)
    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH=dH, cartesian=isinstance(hamiltonian.field, MagneticField_Cartesian))
    Psi_gradK_grad_H = np.matmul(Psi, gradK_grad_H)

    d = np.array(list(dH.values())[:6])
    dq_dtau, dK_dtau = d[:3], d[3:]
    
    dPsi_dtau = (
        - np.matmul(np.matmul(Psi, gradK_gradK_H), Psi)
        - Psi_gradK_grad_H
        - np.transpose(Psi_gradK_grad_H) # - np.matmul(grad_gradK_H, Psi)
        - grad_grad_H
    )

    return pack_beam_parameters(dq_dtau, dK_dtau, dPsi_dtau)



def evolve_beam(
    tau_leave: float,
    tau_points: FloatArray,
    q_initial: FloatArray,
    K_initial: FloatArray,
    Psi_3D_initial_labframe: ComplexFloatArray,
    hamiltonian: Hamiltonian,
    rtol: float,
    atol: float,
) -> Tuple[int, FloatArray, FloatArray, FloatArray, ComplexFloatArray]:
    
    """
    something
    """

    # Packing the ray and beam parameters
    beam_parameters_initial = pack_beam_parameters(q_initial, K_initial, Psi_3D_initial_labframe)

    # Evolving Psi_3D by solving the beam-tracing equation
    log.info(f"Starting the beam solver")

    (   solver_beam_output,
        duration_beam_solver
    ) = timer(solve_ivp)(
        fun=d_beam_parameters_d_tau,
        t_span=[0, tau_leave],
        y0=beam_parameters_initial,
        method="RK45",
        t_eval=tau_points,
        dense_output=False,
        events=None,
        vectorized=False,
        args=(hamiltonian,),
        rtol=rtol,
        atol=atol,
    )

    log.info(f"""\n
        Beam solver status: {solver_beam_output.status}
        Beam solver took {duration_beam_solver} s
        Number of beam evolution evaluations: {solver_beam_output.nfev}
        Time per beam evolution evaluation: {duration_beam_solver / solver_beam_output.nfev}
    """)

    return solver_beam_output.status, solver_beam_output.t, *unpack_beam_parameters(solver_beam_output.y)