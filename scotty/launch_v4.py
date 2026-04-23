import logging
from typing import cast, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.checks_v4 import VALID_FIELDS, VALID_LAUNCH_FLAGS, VALID_LAUNCH_MODE_FLAGS, VALID_BOUNDARY_FLAGS
from scotty.fun_general_v4 import (
    find_K_labframe_cart_to_cyl,
    find_q_labframe_cart_to_cyl,
    find_q_labframe_cyl_to_cart,
    find_Psi_3D_labframe_cart_to_cyl,
    ray_line,
    poloidal_flux_difference_along_ray_line,
    make_array_3x3,
    find_inverse_2D,
)
from scotty.geometry_v4 import MagneticField_Cylindrical, MagneticField_Cartesian
from scotty.hamiltonian_v4 import DielectricTensor, Hamiltonian
from scotty.logger_v4 import arr2str, mln
from scotty.typing import FloatArray

log = logging.getLogger(__name__)

def find_plasma_entry_position(
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    q_launch: FloatArray,
    launch_flag: VALID_LAUNCH_FLAGS,
    field: VALID_FIELDS,
    poloidal_flux_enter: float,
    boundary_adjust: float = 1e-6,
) -> FloatArray:
    
    log.info(f"Finding plasma entry position")

    # If launch_flag is "plasma", we are already in the plasma
    # so take this as the entry position
    if launch_flag == "plasma":
        log.debug(f"`launch_flag` = {launch_flag}, so skipping plasma entry position calculations")
        return q_launch

    # The following calculations will be performed in cartesian and
    # then depending on the geometry we convert accordingly. The following
    # code is equivalent to finding the part of the ray that first enters
    # the plasma and is going deeper. We do this entirely in cartesian
    # coordinates, and then depending on the geometry we convert accordingly.

    if isinstance(field, MagneticField_Cylindrical):
        q_launch_cartesian = find_q_labframe_cyl_to_cart(q_launch)
        max_dist = max(q_launch_cartesian)
    else: # isinstance(field, MagneticField_Cartesian):
        q_launch_cartesian = q_launch
        max_dist = max(q_launch_cartesian)

    # Defining some wrapper functions for ease
    def _ray_line_wrapper(tau): return ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam) # type: ignore
    def _poloidal_flux_difference_along_ray_line_wrapper(tau): return poloidal_flux_difference_along_ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field.polflux_in_cartesian, poloidal_flux_enter) # type: ignore

    # Given a particular field configuration, we find the maximum
    # distance that can be travelled by a ray before 'striking the
    # centre column', then parametrise the path into sections of
    # 0.01m each, and find how many are needed. We also insist that
    # there are at least 100 steps

    num_tau = max(max_dist / 0.01, 100)
    tau_arr = np.linspace(0, max_dist, num_tau)

    # We then launch a ray along this line, which is parametrised by tau,
    # calculate the poloidal flux coordinates along this line, filter out
    # the `NaN`s, find the index of the minimum poloidal flux value, and
    # then create a refined array by re-selecting the start and end taus.
    # If the entire array is `NaN`s, then raise error

    X_arr, Y_arr, Z_arr = ray_line(*q_launch_cartesian, tau_arr, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam) # type: ignore
    polflux_arr = field.polflux_in_cartesian(X_arr, Y_arr, Z_arr)

    if np.isnan(polflux_arr).all(): raise RuntimeError(f"The ray does not intersect the plasma. Check that the launch position is from q_zeta = 0 with acute launch angles")

    start_idx = np.where(~np.isnan(polflux_arr))[0][0]
    stop_idx = np.nanargmin(polflux_arr)

    start_tau = tau_arr[start_idx]
    stop_tau = tau_arr[stop_idx]

    # With the `tau`s of  the start and endpoints, we now create a
    # refined array, get the corresponding poloidal flux values, and
    # feed it into a cubic spline to find the root (i.e. `tau` value
    # where poloidal flux is close to `poloidal_flux_enter`

    tau_arr_refined = np.linspace(start_tau, stop_tau, num_tau)
    polflux_arr_refined = field.polflux_in_cartesian(*ray_line(*q_launch_cartesian, tau_arr_refined, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)) # type: ignore

    if np.isnan(polflux_arr_refined).all(): log.warning(f"NaNs should not occur in the refined `tau` array search")

    spline = CubicSpline(tau_arr_refined, polflux_arr_refined - poloidal_flux_enter, extrapolate=False)
    spline_roots = np.array(spline.roots())
    
    # If there are no roots, then the ray never actually enters the
    # plasma, and we should abort. We also get an idea of where the
    # closest encounter the ray line makes with the plasma is

    if len(spline_roots) == 0:
        minimum = minimize_scalar(_poloidal_flux_difference_along_ray_line_wrapper)
        q_closest_approach_cartesian = _ray_line_wrapper(minimum)
        q_closest_approach_cylindrical = find_q_labframe_cart_to_cyl(q_closest_approach_cartesian)
        _str = f"[R,zeta,Z] = {arr2str(q_closest_approach_cylindrical)}" if isinstance(field, MagneticField_Cylindrical) else f"[X,Y,Z] = {arr2str(q_closest_approach_cartesian)}"
        raise RuntimeError(f"The ray does not intersect the plasma. Closest point is at {_str},"
                           f"distance in poloidal flux to boundary = {minimum.fun}") # type: ignore
    
    # The spline roots are a pretty good guess for the boundary
    # location, which we now try to refine

    root = spline_roots[(np.abs(spline_roots - poloidal_flux_enter)).argmin()]
    boundary = root_scalar(_poloidal_flux_difference_along_ray_line_wrapper, x0=root, x1=root + 1e-3)
    if not boundary.converged: raise RuntimeError(f"Unable to find plasma boundary. Root finding failed with `{boundary.flag}`")
    else: boundary_tau = boundary.root

    # The root might be just outside the plasma due to floating point
    # errors. If so, take a small step of size `boundary_adjust` to
    # ensure the ray is definitely inside

    q_initial_cartesian = _ray_line_wrapper(boundary_tau)
    if field.polflux_in_cartesian(*q_initial_cartesian) > poloidal_flux_enter:
        q_initial_cartesian = _ray_line_wrapper(boundary_tau + boundary_adjust)
        log.debug(f"""
        Poloidal flux at plasma entry position is greater than `poloidal_flux_enter.`
        
        Adjusting by a small tau = {boundary_adjust} to obtain adjusted entry point [X,Y,Z] = {arr2str(q_initial_cartesian)}
        """)

    # Final conversions back to the proper geometry
    log.debug(f"Plasma entry position, `q_initial_cartesian`, is [X,Y,Z] = {arr2str(q_initial_cartesian)}")
    return find_q_labframe_cart_to_cyl(q_initial_cartesian) if isinstance(field, MagneticField_Cylindrical) else q_initial_cartesian



def find_auto_delta_signs(
    auto_delta_sign: bool,
    q_initial: FloatArray,
    deltas: FloatArray,
    field: VALID_FIELDS,
) -> FloatArray:

    # Now, we also perform auto_delta_sign checks (if the user specifies)
    # If the user does not want to flip the signs, just return them
    # Otherwise, check which geometry is being used and check the signs
    if auto_delta_sign: return deltas

    log.info(f"Setting delta signs")
    deltas = np.asarray(deltas, dtype=float)

    def _check_delta_and_log(derivative, delta_name, delta_value):
        if derivative(*q_initial, delta_value) > 0:
            log.debug(f"Switching `{delta_name}` from `{delta_value}` to `{-delta_value}`")
            delta_value = -delta_value
        return delta_value

    if isinstance(field, MagneticField_Cylindrical):
        delta_R, delta_Z = deltas
        delta_R = _check_delta_and_log(field.d_polflux_dR, "delta_R", delta_R)
        delta_Z = _check_delta_and_log(field.d_polflux_dZ, "delta_Z", delta_Z)
        deltas = np.array([delta_R, delta_Z])
    
    else: # isinstance(field, MagneticField_Cartesian):
        delta_X, delta_Y, delta_Z = deltas
        delta_X = _check_delta_and_log(field.d_polflux_dX, "delta_X", delta_X)
        delta_Y = _check_delta_and_log(field.d_polflux_dY, "delta_Y", delta_Y)
        delta_Z = _check_delta_and_log(field.d_polflux_dZ, "delta_Z", delta_Z)
        deltas = np.array([delta_X, delta_Y, delta_Z])
    
    return deltas



def find_plasma_entry_parameters(
    launch_flag: VALID_LAUNCH_FLAGS,
    boundary_flag: VALID_BOUNDARY_FLAGS,
    mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    q_launch: FloatArray,
    q_initial: FloatArray,
    launch_beam_width: float,
    launch_beam_curvature: float,
    field: VALID_FIELDS,
    K_plasmaLaunch_cartesian: FloatArray,
    Psi_3D_plasmaLaunch_labframe_cartesian: FloatArray,
    hamiltonian_pos1: Optional[Hamiltonian] = None,
    hamiltonian_neg1: Optional[Hamiltonian] = None,
    tol_H: float = 1e-5,
    tol_O_mode_polarisation: float = 0.25
) -> Tuple[Union[FloatArray, float, int, None], ...]:
    
    log.info(f"Finding plasma entry parameters")

    log.debug(f"""
        ##################################################
        #
        # Finding plasma entry parameters with:
        #   - launch_flag = {launch_flag}
        #   - boundary_flag = {boundary_flag}
        #   - mode_flag (at launch, from user) = {mode_flag_launch}
        #
        ##################################################
        """)
    
    # The following calculations will be performed in cartesian and
    # then depending on the geometry we convert accordingly. The following
    # code is equivalent to finding the part of the ray that first enters
    # the plasma and is going deeper. We do this entirely in cartesian
    # coordinates, and then depending on the geometry we convert accordingly.
    if isinstance(field, MagneticField_Cylindrical):
        cart = False
        q_launch_cartesian  = find_q_labframe_cyl_to_cart(q_launch)
        q_initial_cartesian = find_q_labframe_cyl_to_cart(q_initial)
    else: # isinstance(field, MagneticField_Cartesian):
        cart = True
        q_launch_cartesian  = q_launch
        q_initial_cartesian = q_initial
    
    # If `launch_flag` = "plasma", that means we start propagation
    # from inside the plasma, so we can skip the plasma entry calculations
    # TODO -- TO REMOVE -- this isnt fully implemented, because we dont
    # select the mode_flag_initial based mode_flag_launch, and we also
    # dont calculate the corresponding e_hat, mode_index, etc
    if launch_flag == "plasma":
        log.debug(f"`launch_flag` = {launch_flag}. Launching directly from inside the plasma")
        K_launch_cartesian = None
        K_initial_cartesian = K_plasmaLaunch_cartesian
        Psi_3D_launch_labframe_cartesian = None
        Psi_3D_entry_labframe_cartesian = None
        Psi_3D_initial_labframe_cartesian = Psi_3D_plasmaLaunch_labframe_cartesian
        distance_from_launch_to_entry = None
        e_hat_initial = None
        mode_flag_initial = mode_flag_launch
        mode_index = None
    
    # Otherwise, we start propagation from vacuum, so we need to
    # calculate the K and Psi, and plus other stuff
    # This is fully implemented -- TO REMOVE this msg
    else:
        log.debug(f"`launch_flag` = {launch_flag}. Launching directly from the plasma boundary")
        poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_deg_Torbeam)
        toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_deg_Torbeam)

        if hamiltonian_pos1 is not None:
            launch_angular_freq = hamiltonian_pos1.angular_frequency
            K0 = hamiltonian_pos1.wavenumber
            density_fit = hamiltonian_pos1.density
            temperature_fit = hamiltonian_pos1.temperature
        
        elif hamiltonian_neg1 is not None:
            launch_angular_freq = hamiltonian_neg1.angular_frequency
            K0 = hamiltonian_neg1.wavenumber
            density_fit = hamiltonian_neg1.density
            temperature_fit = hamiltonian_neg1.temperature
        
        else: raise ValueError(f"At least one of `hamiltonian_pos1` and `hamiltonian_neg1` must be provided to find the plasma entry parameters")
        
        K0 = cast(float, K0) # to stop the typechecker from complaining
        
        # Find K_launch
        K_launch_cartesian = -K0 * np.array([np.cos(poloidal_launch_angle) * np.cos(toroidal_launch_angle),
                                             np.cos(poloidal_launch_angle) * np.sin(toroidal_launch_angle),
                                             np.sin(poloidal_launch_angle)])
        
        # Finding Psi_w_launch_beamframe_cartesian and Psi_3D_launch_beamframe_cartesian
        # Entries on the off-diagonal = 0, because beamframe
        # Entries on the diagonal = K_0/R + 2i/W^2, where:
        #    R is beam radius of curvature (in metres); and
        #    W is beam width (in metres)
        # First row/column is y-direction; second is x-direction; third is g-direction (beamframe)
        # Not to be confused with X-, Y-, Z-directions (labframe)
        diag = K0*launch_beam_curvature + 2j/launch_beam_width**2
        Psi_w_launch_beamframe_cartesian = diag * np.eye(2)
        Psi_3D_launch_beamframe_cartesian = make_array_3x3(Psi_w_launch_beamframe_cartesian)

        # Setting up the rotation matrices, so that we can convert
        # Psi_3D_launch_beamframe_cartesian into Psi_3D_launch_labframe_cartesian
        poloidal_rotation_angle = poloidal_launch_angle + np.pi/2
        toroidal_rotation_angle = toroidal_launch_angle
        sin_pol, cos_pol = np.sin(poloidal_rotation_angle), np.cos(poloidal_rotation_angle)
        sin_tor, cos_tor = np.sin(toroidal_rotation_angle), np.cos(toroidal_rotation_angle)
        poloidal_rotation_matrix = np.array([[ cos_pol,       0, sin_pol],
                                            [       0,       1,       0],
                                            [-sin_pol,       0, cos_pol]])
        toroidal_rotation_matrix = np.array([[ cos_tor, sin_tor,       0],
                                            [-sin_tor, cos_tor,       0],
                                            [       0,       0,       1]])
        rotation_matrix = np.matmul(poloidal_rotation_matrix, toroidal_rotation_matrix)
        rotation_matrix_inverse = np.transpose(rotation_matrix)

        # Finding Psi_3D_launch_labframe_cartesian using:
        # Psi_labframe = R^-1 * Psi_beamframe * R, where
        #    R is the rotation matrix to convert a vector from beamframe to labframe
        Psi_3D_launch_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_launch_beamframe_cartesian, rotation_matrix))

        # Now we propagate the beam until it reaches the plasma boundary,
        # and then apply either the continuous or discontinuous or no
        # boundary conditions to find K_entry and Psi_entry when the
        # beam enters the plasma
        Psi_w_inverse_launch_beamframe_cartesian = find_inverse_2D(Psi_w_launch_beamframe_cartesian)
        distance_from_launch_to_entry = np.linalg.norm(q_launch_cartesian - q_initial_cartesian)
        Psi_w_inverse_entry_beamframe_cartesian = distance_from_launch_to_entry / K0 * np.eye(2) + Psi_w_inverse_launch_beamframe_cartesian

        # 'Psi_3D_entry' is still in vacuum, so the components of Psi in the
        # beam frame along g are all zero (since grad_H = 0)
        Psi_3D_entry_beamframe_cartesian = make_array_3x3(find_inverse_2D(Psi_w_inverse_entry_beamframe_cartesian))
        Psi_3D_entry_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_entry_beamframe_cartesian, rotation_matrix))

        # If `boundary_flag` is None, then we assume that the electron density
        # profile is both continuous and differentiable at the plasma boundary,
        # so we can just take the values at the plasma launch position as the
        # values to start the solvers at
        # TODO -- TO REMOVE -- not fully implemented, because we dont select
        # the mode_flag_initial, the mode_index, and the e_hat_initial
        if boundary_flag is None:
            log.debug(f"`boundary_flag` = {boundary_flag}. No boundary conditions applied")
            K_initial_cartesian = K_launch_cartesian
            Psi_3D_initial_labframe_cartesian = Psi_3D_entry_labframe_cartesian
            e_hat_initial = None # TO REMOVE -- not implemented yet
            mode_flag_initial = mode_flag_launch # TO REMOVE -- not implemented yet
            mode_index = None # TO REMOVE -- not implemented yet
        
        # If `boundary_flag` = "continuous" or "discontinuous", then we assume
        # that the electron density is not differentiable at the plasma boundary, 
        # so we apply the appropriate boundary condition calculations. Here, we use
        # K_entry and Psi_entry (without boundary conditions) and get K_initial and
        # Psi_initial (with boundary conditions) which are later fed into the
        # `solve_ivp` as the initial values. We solve this for mode_flags = +1 and -1,
        # then later we check to see which set of K_initial and Psi_initial
        # corresponds to O or X mode
        else:
            log.debug(f"`boundary_flag` = {boundary_flag}. Applying {boundary_flag} boundary conditions to find K and Psi")
            polflux = field.polflux_in_cartesian(*q_initial_cartesian)
            electron_density = density_fit(polflux)
            electron_temperature = temperature_fit(polflux) if temperature_fit else None

            B_magnitude = field.magnitude(*q_initial)
            b_hat_cartesian = field.unitvector_in_cartesian(*q_initial)
            epsilon = DielectricTensor(launch_angular_freq, B_magnitude, electron_density, electron_temperature)

            log.debug(f"""
        ##################################################
        #
        # Calculated values at [X,Y,Z] = {arr2str(q_initial_cartesian)}:
        #   - poloidal flux = {polflux}
        #   - n_e = {electron_density} (e19)
        #   - T_e = {electron_temperature}
        #   - |B| = {B_magnitude}
        #   - b_hat_cartesian = {arr2str(b_hat_cartesian)}
        #   - epsilon_11 = {epsilon.e_11}
        #   - epsilon_12 = {epsilon.e_12}
        #   - epsilon_bb = {epsilon.e_bb}
        #
        ##################################################
        """)
            
            # If mode_flag is 1 or -1, then we just calculate the
            # quantities for those. If mode_flag is "O" or "X", then
            # we calculate the quantities for both 1 and -1. After
            # obtaining the quantities corresponding to H_booker = 0,
            # we calculate H_Cardano and find the polarisation vector
            # corresponding to H_booker = H_Cardano = 0 to see if it's
            # O- or X-mode

            O_mode, X_mode = [], []


















    
    # If `launch_flag` = "vacuum", that means we start propagation
    # from right at the plasma boundary
    # TODO -- TO REMOVE -- this isnt fully implemented, because we dont
    # select the mode_flag_initial based mode_flag_launch, and we also
    # dont calculate the corresponding e_hat, mode_index, etc
    elif launch_flag == "vacuum":
        log.debug(f"`launch_flag` = {launch_flag}. Launching directly from the plasma boundary")

    
    return

    # alr declared, need to return

    # K_launch_cartesian
    # K_initial_cartesian, for None only

    # Psi_3D_launch_labframe_cartesian
    # Psi_3D_entry_labframe_cartesian
    # Psi_3D_initial_labframe_cartesian, for None only

    # distance_from_launch_to_entry



        ##################################################
        #
        # Calculated values at {q_initial_cartesian}:
        #   - K_launch_cartesian = {K_launch_cartesian}
        #   - K_initial_cartesian = {K_initial_cartesian}
        #   - Psi_3D_launch_labframe_cartesian =
        #        {Psi_3D_launch_labframe_cartesian}
        #   - Psi_3D_entry_labframe_cartesian =
        #        {Psi_3D_entry_labframe_cartesian}
        #   - Psi_3D_initial_labframe_cartesian =
        #        {Psi_3D_initial_labframe_cartesian}
        #   - distance_from_launch_to_entry = {distance_from_launch_to_entry}
        #   - e_hat_initial = {e_hat_initial}
        #   - mode_flag at plasma boundary (corresponding to mode_flag at launch) = {mode_flag_initial}
        #   - mode_index = {mode_index}
        #
        ##################################################
        """)