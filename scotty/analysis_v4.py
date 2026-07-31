import logging
import numpy as np
from scotty.checks_v4 import VALID_FIELDS
from scotty.hamiltonian_v4 import Hamiltonian
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
    
    tau_array = np.array(solver_output.tau)

    # Position and wavevectors
    q_vec = np.array(solver_output.q_vector)
    K_vec = np.array(solver_output.K_vec)









def dbs_analysis():

    log.info(f"""\n
        ##################################################
        #
        # DOPPLER BACKSCATTERING ANALYSIS ROUTINE
        #
        ##################################################
        """)