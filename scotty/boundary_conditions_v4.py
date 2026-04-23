import logging
import numpy as np
from scotty.checks_v4 import VALID_FIELDS, MagneticField_Cylindrical, MagneticField_Cartesian
from scotty.fun_general_v4 import find_vector_and_q_cyl_to_cart, find_q_labframe_cart_to_cyl, find_K_labframe_cart_to_cyl, find_Psi_3D_labframe_cart_to_cyl
from scotty.hamiltonian_v4 import Hamiltonian
from scotty.logger_v4 import arr2str
from scotty.typing import FloatArray
from typing import Tuple

log = logging.getLogger(__name__)

##################################################
#
# BOUNDARY CONDITION CHECKS
#
##################################################

def check_vector_pointing_into_plasma(q: FloatArray, vector: FloatArray, field: VALID_FIELDS) -> None:
    if not q.shape == (3,): raise ValueError(f"`q` must have shape (3,), but got {q.shape}")
    if not vector.shape == (3,): raise ValueError(f"`vector` must have shape (3,), but got {vector.shape}")

    if isinstance(field, MagneticField_Cylindrical): v_XYZ, q_XYZ = find_vector_and_q_cyl_to_cart(vector, q)
    else: v_XYZ, q_XYZ = vector, q

    unitv_XYZ = v_XYZ / np.linalg.norm(v_XYZ)

    q_XYZ_minus = q_XYZ - 0.01*unitv_XYZ
    q_XYZ_plus  = q_XYZ + 0.01*unitv_XYZ

    polflux_at_q_minus = field.polflux_in_cartesian(*q_XYZ_minus)
    polflux_at_q =       field.polflux_in_cartesian(*q_XYZ)
    polflux_at_q_plus =  field.polflux_in_cartesian(*q_XYZ_plus)

    if polflux_at_q_minus < polflux_at_q < polflux_at_q_plus: raise ValueError(f"K_plasma is pointing out of the plasma!")
    elif polflux_at_q_plus < polflux_at_q < polflux_at_q_minus: pass
    else: log.warning(f"Warning: Unable to check if K_plasma is pointing in or out of the plasma!")

##################################################
#
# BOUNDARY CONDITION CODES
#
##################################################

def apply_continuous_BC(
    q_entry_cartesian: FloatArray,
    K_entry_cartesian: FloatArray,
    Psi_3D_entry_labframe_cartesian: FloatArray,
    field: VALID_FIELDS,
    hamiltonian: Hamiltonian,
) -> Tuple[FloatArray, FloatArray]:
    r"""Apply boundary conditions at the plasma-vacuum boundary
    when the electron density is continuous but not differentiable,
    i.e. continuous n_e but discontinuous grad(n_e)

    In the continuous boundary condition case, the continuity of
    the electron density means that K_plasma = K_vacuum, and thus
    we only concern ourselves with finding Psi_3D in the plasma
    """

    log.debug(f"Applying continuous boundary conditions")

    # Getting important quantities
    # Note that all calculations, unlike in `launch.py`, are done
    # in their own coordinate systems (for abstraction purposes)
    # In these coordinate systems, first get the flux derivatives
    # which will be used for the interface matrix later
    if isinstance(field, MagneticField_Cylindrical):
        cart = False
        q_entry = find_q_labframe_cart_to_cyl(q_entry_cartesian)
        K_entry = find_K_labframe_cart_to_cyl(K_entry_cartesian, q_entry_cartesian)
        Psi_entry = find_Psi_3D_labframe_cart_to_cyl(Psi_3D_entry_labframe_cartesian, K_entry_cartesian, q_entry_cartesian)
        delta_R, delta_Z = hamiltonian.spacings["R"], hamiltonian.spacings["Z"]
        dp = {
            "d(polflux)_dR": field.d_polflux_dR(*q_entry, delta_R=delta_R),
            "d(polflux)_dzeta": np.array([0.0]),
            "d(polflux)_dZ": field.d_polflux_dZ(*q_entry, delta_Z=delta_Z),
        }
    else:
        cart = True
        q_entry = q_entry_cartesian
        K_entry = K_entry_cartesian
        Psi_entry = Psi_3D_entry_labframe_cartesian
        delta_X, delta_Y, delta_Z = hamiltonian.spacings["X"], hamiltonian.spacings["Y"], hamiltonian.spacings["Z"]
        dp = {
            "d(polflux)_dX": field.d_polflux_dX(*q_entry, delta_X=delta_X),
            "d(polflux)_dY": field.d_polflux_dY(*q_entry, delta_Y=delta_Y),
            "d(polflux)_dZ": field.d_polflux_dZ(*q_entry, delta_Z=delta_Z),
        }
    
    # Now get the spatial and wavevector derivatives. Note
    # that, even for abstraction purposes, we use the spatial
    # coordinates X, Y, Z and their conjugate wavevectors
    # Kx, Ky, Kz even in the cylindrical coordinate system
    # because the underlying calculations are essentially the
    # same, save for the derivatives in zeta which are equal
    # to zero due to symmetry considerations
    dH = hamiltonian.derivatives(q_entry, K_entry)
    dH_dX, dH_dY, dH_dZ, dH_dKx, dH_dKy, dH_dKz = dH.values()
    dH.update(dp)
    dp_dX, dp_dY, dp_dZ = dp.values()

    _printmsg = "\n".join(f"        #   - {k} = {v}" for k, v in dH.items())
    log.debug(f"""
        Finding Psi at the plasma entry point with continuous boundary conditions
        ##################################################
        # Derivatives at {"[X, Y, Z]" if cart else "[R, zeta, Z]"} = {q_entry} with {"[K_X, K_Y, K_Z]" if cart else "[K_R, K_zeta, K_Z]"} = {K_entry}: \n{_printmsg}
        #""")

    # Gradients at the plasma-vacuum boundary can be finnicky,
    # so best to check
    for derivative in dH.keys():
        if np.isnan(dH[derivative]):
            raise ValueError(f"{derivative} is NaN")
        
    # At the plasma-vacuum boundary, we have two Psi matrices:
    # one corresponding to Psi in the vacuum (entry), and the
    # other corresponding to Psi in the plasma (initial). We
    # denote these by the subscripts 'v' and 'p' respectively
    Psi_XX_v = Psi_entry[0,0]
    Psi_XY_v = Psi_entry[0,1]
    Psi_XZ_v = Psi_entry[0,2]
    Psi_YY_v = Psi_entry[1,1]
    Psi_YZ_v = Psi_entry[1,2]
    Psi_ZZ_v = Psi_entry[2,2]

    # Now we set up the interface matrix using 6 linearly
    # independent equations to obtain a relation between the
    # entries of "Psi_v" and "Psi_p"
    interface_matrix = np.array([
        [dp_dY**2, -2*dp_dX*dp_dY,  0,                       dp_dX**2,  0,                       0                 ],
        [dp_dZ**2,  0,             -2*dp_dX*dp_dZ,           0,         0,                       dp_dX**2          ],
        [dp_dZ**2,  2*dp_dZ**2,    -2*dp_dZ*(dp_dX + dp_dY), dp_dZ**2, -2*dp_dZ*(dp_dX + dp_dY), (dp_dX + dp_dY)**2],
        [dH_dKx,    dH_dKy,        dH_dKz,                   0,         0,                       0                 ],
        [0,         dH_dKx,        0,                        dH_dKy,    dH_dKz,                  0                 ],
        [0,         0,             dH_dKx,                   0,         dH_dKy,                  dH_dKz            ],
    ], dtype=np.float64)

    log.debug(f"""
        #
        #   - interface matrix =
        #        {arr2str(interface_matrix[0])}
        #        {arr2str(interface_matrix[1])}
        #        {arr2str(interface_matrix[2])}
        #        {arr2str(interface_matrix[3])}
        #        {arr2str(interface_matrix[4])}
        #        {arr2str(interface_matrix[5])}
        #""")
    
    # Comment from the original function code:
        # interface_matrix will be singular if one tries to
        # transition while still in vacuum (and there's no
        # plasma at all); at least that's what happens in
        # my experience
    try: interface_matrix_inverse = np.linalg.inv(interface_matrix)
    except np.linalg.LinAlgError as e: raise np.linalg.LinAlgError(f"Singular matrix when calculating `interface_matrix_inverse`. This is caused when `interface_matrix` is singular, usually occuring when one tries to apply boundary conditions and 'transition' from vacuum to plasma while actually still in vacuum (and there's no plasma at all)") from e

    log.debug(f"""
        #
        #   - interface matrix inverse =
        #        {arr2str(interface_matrix_inverse[0])}
        #        {arr2str(interface_matrix_inverse[1])}
        #        {arr2str(interface_matrix_inverse[2])}
        #        {arr2str(interface_matrix_inverse[3])}
        #        {arr2str(interface_matrix_inverse[4])}
        #        {arr2str(interface_matrix_inverse[5])}
        #""")
    
    # For continuous boundary conditions, we do not
    # calculate the `eta`s
    log.debug(f"""
        #
        #   - eta_XY  = None
        #   - eta_XZ  = None
        #   - eta_XYZ = None
        #""")

    

    


    return