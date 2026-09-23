import logging
import numpy as np
from scotty.checks_v4 import VALID_FIELDS
from scotty.hamiltonian_v4 import Hamiltonian, convert_hessians, hessians
from scotty.profile_fit import ProfileFitLike
from typing import Optional
import xarray as xr

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]

def basic_analysis(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    hamiltonian: Hamiltonian,
    hamiltonian_other: Hamiltonian,
    field: VALID_FIELDS,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike]):

    log.info(f"""\n
        ##################################################
        #
        # BASIC ANALYSIS ROUTINE
        #
        ##################################################
        """)

    log.debug(f"Performing analysis on ray-tracing results")

    cart = bool(inputs["geometry"] == "cartesian")
    btf = not bool(inputs["ray_tracing_flag"])

    tau = np.array(solver_output["tau"][()])
    len_tau = len(tau)

    # Position and wavevector stuff
    q_vec_cart = np.array(solver_output["q_output_cartesian"][()]).T   # (N,3) -> (3,N)
    q_vec_cyld = np.array(solver_output["q_output_cylindrical"][()]).T # (N,3) -> (3,N)
    K_vec_cart = np.array(solver_output["K_output_cartesian"][()]).T   # (N,3) -> (3,N)
    K_vec_cyld = np.array(solver_output["K_output_cylindrical"][()]).T # (N,3) -> (3,N)

    # Booker Hamiltonian stuff
    if cart: q_vec, K_vec = q_vec_cart, K_vec_cart
    else:    q_vec, K_vec = q_vec_cyld, K_vec_cyld
    H_Booker = hamiltonian(**q_vec, **K_vec) # type: ignore
    H_Booker_other = hamiltonian_other(**q_vec, **K_vec) # type: ignore
    dH = hamiltonian.derivatives(q_vec, K_vec, second_order=btf)
    if btf:
        _temp = hessians(dH, cartesian=cart)
        if cart: grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = _temp
        else:
            grad_grad_H_cyld, gradK_grad_H_cyld, gradK_gradK_H_cyld = _temp
            grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = convert_hessians(
                q_start = q_vec_cyld,
                dH = dH,
                grad_grad_H_start = grad_grad_H_cyld,
                gradK_grad_H_start = gradK_grad_H_cyld,
                gradK_gradK_H_start = gradK_gradK_H_cyld,
                start = "cylindrical",
                end = "cartesian",
            )

    # Finite difference spacings
    # delta



    # find_g_cartesian





    #
    polflux = field.polflux_in_cartesian(**q_vec_cart) # type: ignore






    
    









def dbs_analysis():

    log.info(f"""\n
        ##################################################
        #
        # DOPPLER BACKSCATTERING ANALYSIS ROUTINE
        #
        ##################################################
        """)