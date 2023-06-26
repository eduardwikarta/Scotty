from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import constants
from scipy.integrate import cumtrapz
import xarray as xr

from scotty.check_output import check_output
from scotty.profile_fit import ProfileFitLike
from scotty.derivatives import derivative
from scotty.fun_general import (
    K_magnitude,
    contract_special,
    find_nearest,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    find_q_lab_Cartesian,
    make_unit_vector_from_cross_product,
    find_Psi_3D_lab_Cartesian,
    find_H_Cardano,
    angular_frequency_to_wavenumber,
    find_waist,
    find_D,
    find_x0,
    find_electron_mass,
)
from scotty.geometry import MagneticField
from scotty.hamiltonian import DielectricTensor, Hamiltonian, hessians
from scotty.typing import FloatArray


def save_npz(filename: Path, df: xr.Dataset) -> None:
    """Save xarray dataset to numpy .npz file"""
    np.savez(
        filename,
        **{str(k): v for k, v in df.items()},
        **{str(k): v.data for k, v in df.coords.items()},
    )


def immediate_analysis(
    solver_output: xr.Dataset,
    field: MagneticField,
    find_density_1D: ProfileFitLike,
    find_temperature_1D: Optional[ProfileFitLike],
    hamiltonian: Hamiltonian,
    K_zeta_initial: float,
    launch_angular_frequency: float,
    mode_flag: int,
    delta_R: float,
    delta_Z: float,
    delta_K_R: float,
    delta_K_zeta: float,
    delta_K_Z: float,
    Psi_3D_lab_launch: FloatArray,
    Psi_3D_lab_entry: FloatArray,
    distance_from_launch_to_entry: float,
    vacuumLaunch_flag: bool,
    output_path: Path,
    output_filename_suffix: str,
    dH: Dict[str, FloatArray],
):
    q_R = solver_output.q_R
    q_Z = solver_output.q_Z
    tau = solver_output.tau
    K_R = solver_output.K_R
    K_Z = solver_output.K_Z

    numberOfDataPoints = len(tau)

    poloidal_flux = field.poloidal_flux(q_R, q_Z)

    dH_dR = dH["dH_dR"]
    dH_dZ = dH["dH_dZ"]
    dH_dKR = dH["dH_dKR"]
    dH_dKzeta = dH["dH_dKzeta"]
    dH_dKZ = dH["dH_dKZ"]

    # Calculates nabla_K H
    # Calculates g_hat
    g_hat = np.zeros([numberOfDataPoints, 3])
    g_magnitude = (q_R**2 * dH_dKzeta**2 + dH_dKR**2 + dH_dKZ**2) ** 0.5
    g_hat[:, 0] = dH_dKR / g_magnitude  # g_hat_R
    g_hat[:, 1] = q_R * dH_dKzeta / g_magnitude  # g_hat_zeta
    g_hat[:, 2] = dH_dKZ / g_magnitude  # g_hat_Z

    # Calculates b_hat and grad_b_hat
    B_R = field.B_R(q_R, q_Z)
    B_T = field.B_T(q_R, q_Z)
    B_Z = field.B_Z(q_R, q_Z)
    B_magnitude = field.magnitude(q_R, q_Z)
    b_hat = field.unit(q_R, q_Z)

    dbhat_dR = derivative(
        field.unit, dims="q_R", args={"q_R": q_R, "q_Z": q_Z}, spacings=delta_R
    )
    dbhat_dZ = derivative(
        field.unit, dims="q_Z", args={"q_R": q_R, "q_Z": q_Z}, spacings=delta_Z
    )

    # Transpose dbhat_dR so that it has the right shape
    grad_bhat = np.zeros([numberOfDataPoints, 3, 3])
    grad_bhat[:, 0, :] = dbhat_dR
    grad_bhat[:, 2, :] = dbhat_dZ
    grad_bhat[:, 1, 0] = -B_T / (B_magnitude * q_R)
    grad_bhat[:, 1, 1] = B_R / (B_magnitude * q_R)

    # x_hat and y_hat
    y_hat = make_unit_vector_from_cross_product(b_hat, g_hat)
    x_hat = make_unit_vector_from_cross_product(y_hat, g_hat)

    # -------------------
    # Not useful for physics or data analysis
    # But good for checking whether things are working properly
    # -------------------
    #
    H = hamiltonian(q_R.data, q_Z.data, K_R.data, K_zeta_initial, K_Z.data)
    # Create and immediately evaluate a Hamiltonian with the opposite mode
    H_other = Hamiltonian(
        field,
        launch_angular_frequency,
        -mode_flag,
        find_density_1D,
        delta_R,
        delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
    )(q_R.data, q_Z.data, K_R.data, K_zeta_initial, K_Z.data)

    # -------------------
    # Sanity check. Makes sure that calculated quantities are reasonable
    # -------------------
    check_output(H)

    df = xr.Dataset(
        {
            "B_R": (["tau"], B_R),
            "B_T": (["tau"], B_T),
            "B_Z": (["tau"], B_Z),
            "B_magnitude": (["tau"], B_magnitude),
            "K_R": K_R,
            "K_Z": K_Z,
            "K_zeta_initial": K_zeta_initial,
            "Psi_3D_lab_launch": (["row", "col"], Psi_3D_lab_launch),
            "Psi_3D": solver_output.Psi_3D,
            "b_hat": (["tau", "col"], b_hat),
            "dH_dKR": (["tau"], dH_dKR),
            "dH_dKZ": (["tau"], dH_dKZ),
            "dH_dKzeta": (["tau"], dH_dKzeta),
            "dH_dR": (["tau"], dH_dR),
            "dH_dZ": (["tau"], dH_dZ),
            "g_hat": (["tau", "col"], g_hat),
            "g_magnitude": g_magnitude,
            "grad_bhat": (["tau", "row", "col"], grad_bhat),
            "q_R": q_R,
            "q_Z": q_Z,
            "q_zeta": solver_output.q_zeta,
            "x_hat": (["tau", "col"], x_hat),
            "y_hat": (["tau", "col"], y_hat),
            "poloidal_flux": (["tau"], poloidal_flux),
        },
        coords={"tau": tau, "row": ["R", "zeta", "Z"], "col": ["R", "zeta", "Z"]},
    )

    temperature = find_temperature_1D(poloidal_flux) if find_temperature_1D else None
    if temperature is not None:
        df.update({"temperature": (["tau"], temperature)})

    if vacuumLaunch_flag:
        electron_density = np.asfarray(find_density_1D(poloidal_flux))
        epsilon = DielectricTensor(
            electron_density, launch_angular_frequency, B_magnitude, temperature
        )

        # Plasma and cyclotron frequencies
        normalised_plasma_freqs = find_normalised_plasma_freq(
            electron_density, launch_angular_frequency, temperature
        )
        normalised_gyro_freqs = find_normalised_gyro_freq(
            B_magnitude, launch_angular_frequency, temperature
        )

        vacuum_only = {
            "Psi_3D_lab_entry": (["row", "col"], Psi_3D_lab_entry),
            "distance_from_launch_to_entry": distance_from_launch_to_entry,
            "dpolflux_dR_debugging": (
                ["tau"],
                field.d_poloidal_flux_dR(q_R, q_Z, delta_R),
            ),
            "dpolflux_dZ_debugging": (
                ["tau"],
                field.d_poloidal_flux_dZ(q_R, q_Z, delta_Z),
            ),
            "epsilon_para": (["tau"], epsilon.e_bb),
            "epsilon_perp": (["tau"], epsilon.e_11),
            "epsilon_g": (["tau"], epsilon.e_12),
            "electron_density": (["tau"], electron_density),
            "normalised_plasma_freqs": (["tau"], normalised_plasma_freqs),
            "normalised_gyro_freqs": (["tau"], normalised_gyro_freqs),
            "H": (["tau"], H),
            "H_other": (["tau"], H_other),
        }
        df.update(vacuum_only)

    save_npz(output_path / f"data_output{output_filename_suffix}", df)

    return df


def further_analysis(
    inputs: xr.Dataset,
    df: xr.Dataset,
    Psi_3D_lab_entry_cartersian: FloatArray,
    output_path: Path,
    output_filename_suffix: str,
    field: MagneticField,
    detailed_analysis_flag: bool,
    dH: Dict[str, FloatArray],
):
    # Calculates various useful stuff
    [q_X, q_Y, _] = find_q_lab_Cartesian([df.q_R, df.q_zeta, df.q_Z])
    point_spacing = np.sqrt(
        (np.diff(q_X)) ** 2 + (np.diff(q_Y)) ** 2 + (np.diff(df.q_Z)) ** 2
    )
    distance_along_line = np.cumsum(point_spacing)
    distance_along_line = np.append(0, distance_along_line)
    RZ_point_spacing = np.sqrt((np.diff(df.q_Z)) ** 2 + (np.diff(df.q_R)) ** 2)
    RZ_distance_along_line = np.cumsum(RZ_point_spacing)
    RZ_distance_along_line = np.append(0, RZ_distance_along_line)

    # Calculates the index of the minimum magnitude of K
    # That is, finds when the beam hits the cut-off
    K_magnitude_array = np.asfarray(
        K_magnitude(df.K_R, df.K_zeta_initial, df.K_Z, df.q_R)
    )

    # Index of the cutoff, at the minimum value of K, use this with other arrays
    cutoff_index = find_nearest(np.abs(K_magnitude_array), 0)

    # Calcuating the angles theta and theta_m
    # B \cdot K / (abs (B) abs(K))
    sin_theta_m_analysis = (
        df.b_hat.sel(col="R") * df.K_R
        + df.b_hat.sel(col="zeta") * df.K_zeta_initial / df.q_R
        + df.b_hat.sel(col="Z") * df.K_Z
    ) / K_magnitude_array

    # Assumes the mismatch angle is never smaller than -90deg or bigger than 90deg
    theta_m = np.sign(sin_theta_m_analysis) * np.arcsin(abs(sin_theta_m_analysis))

    kperp1_hat = make_unit_vector_from_cross_product(df.y_hat, df.b_hat)
    # The negative sign is there by definition
    sin_theta_analysis = -contract_special(df.x_hat, kperp1_hat)
    # The negative sign is there by definition. Alternative way to get sin_theta
    # Assumes theta is never smaller than -90deg or bigger than 90deg
    theta = np.sign(sin_theta_analysis) * np.arcsin(abs(sin_theta_analysis))

    cos_theta_analysis = np.cos(theta)
    tan_theta_analysis = np.tan(theta)
    # -----

    # Calcuating the corrections to make M from Psi
    # Includes terms small in mismatch

    # The dominant value of kperp1 that is backscattered at every point
    k_perp_1_bs = -2 * K_magnitude_array * np.cos(theta_m + theta) / cos_theta_analysis

    # Converting x_hat, y_hat, and Psi_3D to Cartesians so we can contract them with each other
    cos_q_zeta = np.cos(df.q_zeta)
    sin_q_zeta = np.sin(df.q_zeta)

    def to_Cartesian(array):
        cart = np.empty([len(df.tau), 3])
        cart[:, 0] = array[:, 0] * cos_q_zeta - array[:, 1] * sin_q_zeta
        cart[:, 1] = array[:, 0] * cos_q_zeta + array[:, 1] * sin_q_zeta
        cart[:, 2] = array[:, 2]
        return cart

    y_hat_Cartesian = to_Cartesian(df.y_hat)
    x_hat_Cartesian = to_Cartesian(df.x_hat)
    g_hat_Cartesian = to_Cartesian(df.g_hat)

    Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(
        df.Psi_3D, df.q_R, df.q_zeta, df.K_R, df.K_zeta_initial
    )
    Psi_xx = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, x_hat_Cartesian)
    )
    Psi_xy = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_yy = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_xg = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_yg = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_gg = contract_special(
        g_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )

    Psi_xx_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, x_hat_Cartesian[0, :]),
    )
    Psi_xy_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )
    Psi_yy_entry = np.dot(
        y_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )

    numberOfDataPoints = len(df.tau)
    # Calculating intermediate terms that are needed for the corrections in M
    xhat_dot_grad_bhat = contract_special(df.x_hat, df.grad_bhat)
    yhat_dot_grad_bhat = contract_special(df.y_hat, df.grad_bhat)
    ray_curvature_kappa = np.zeros([numberOfDataPoints, 3])
    ray_curvature_kappa[:, 0] = (1 / df.g_magnitude) * (
        np.gradient(df.g_hat[:, 0], df.tau)
        - df.g_hat[:, 1] * df.dH_dKzeta  # See notes 07 June 2021
    )
    ray_curvature_kappa[:, 1] = (1 / df.g_magnitude) * (
        np.gradient(df.g_hat[:, 1], df.tau)
        + df.g_hat[:, 0] * df.dH_dKzeta  # See notes 07 June 2021
    )
    ray_curvature_kappa[:, 2] = (1 / df.g_magnitude) * np.gradient(
        df.g_hat[:, 2], df.tau
    )
    kappa_magnitude = np.linalg.norm(ray_curvature_kappa, axis=-1)
    d_theta_d_tau = np.gradient(theta, df.tau)
    d_xhat_d_tau = np.zeros([numberOfDataPoints, 3])
    d_xhat_d_tau[:, 0] = (
        np.gradient(df.x_hat[:, 0], df.tau) - df.x_hat[:, 1] * df.dH_dKzeta
    )  # See notes 07 June 2021
    d_xhat_d_tau[:, 1] = (
        np.gradient(df.x_hat[:, 1], df.tau) + df.x_hat[:, 0] * df.dH_dKzeta
    )  # See notes 07 June 2021
    d_xhat_d_tau[:, 2] = np.gradient(df.x_hat[:, 2], df.tau)

    xhat_dot_grad_bhat_dot_xhat = contract_special(xhat_dot_grad_bhat, df.x_hat)
    xhat_dot_grad_bhat_dot_yhat = contract_special(xhat_dot_grad_bhat, df.y_hat)
    xhat_dot_grad_bhat_dot_ghat = contract_special(xhat_dot_grad_bhat, df.g_hat)
    yhat_dot_grad_bhat_dot_xhat = contract_special(yhat_dot_grad_bhat, df.x_hat)
    yhat_dot_grad_bhat_dot_yhat = contract_special(yhat_dot_grad_bhat, df.y_hat)
    yhat_dot_grad_bhat_dot_ghat = contract_special(yhat_dot_grad_bhat, df.g_hat)
    kappa_dot_xhat = contract_special(ray_curvature_kappa, df.x_hat)
    kappa_dot_yhat = contract_special(ray_curvature_kappa, df.y_hat)
    # This should be 0. Good to check.
    kappa_dot_ghat = contract_special(ray_curvature_kappa, df.g_hat)
    d_xhat_d_tau_dot_yhat = contract_special(d_xhat_d_tau, df.y_hat)

    # Calculates the components of M_w, only taking into consideration
    # correction terms that are not small in mismatch
    M_xx = Psi_xx + (k_perp_1_bs / 2) * xhat_dot_grad_bhat_dot_ghat
    M_xy = Psi_xy + (k_perp_1_bs / 2) * yhat_dot_grad_bhat_dot_ghat
    M_yy = Psi_yy
    # -----

    # Calculates the localisation, wavenumber resolution, and mismatch attenuation pieces
    det_M_w_analysis = M_xx * M_yy - M_xy**2
    M_w_inv_xx = M_yy / det_M_w_analysis
    M_w_inv_xy = -M_xy / det_M_w_analysis
    M_w_inv_yy = M_xx / det_M_w_analysis

    delta_k_perp_2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy))
    delta_theta_m = np.sqrt(
        np.imag(M_w_inv_yy)
        / ((np.imag(M_w_inv_xy)) ** 2 - np.imag(M_w_inv_xx) * np.imag(M_w_inv_yy))
    ) / (K_magnitude_array)
    loc_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

    print("polflux: ", df.poloidal_flux[cutoff_index])

    print("theta_m", theta_m[cutoff_index])
    print("delta_theta_m", delta_theta_m[cutoff_index])
    print(
        "mismatch attenuation",
        np.exp(-2 * (theta_m[cutoff_index] / delta_theta_m[cutoff_index]) ** 2),
    )

    # This part is used to make some nice plots when post-processing
    R_midplane_points = np.linspace(field.R_coord[0], field.R_coord[-1], 1000)
    # poloidal flux at R and z=0
    poloidal_flux_on_midplane = field.poloidal_flux(R_midplane_points, 0)

    # Calculates localisation (start)
    # Ray piece of localisation as a function of distance along ray

    H_1_Cardano, H_2_Cardano, H_3_Cardano = find_H_Cardano(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df.epsilon_para.data,
        df.epsilon_perp.data,
        df.epsilon_g.data,
        theta_m.data,
    )

    def H_cardano(K_R, K_zeta, K_Z):
        # In my experience, the H_3_Cardano expression corresponds to
        # the O mode, and the H_2_Cardano expression corresponds to
        # the X-mode.

        # ALERT: This may not always be the case! Check the output
        # figure to make sure that the appropriate solution is indeed
        # 0 along the ray
        result = find_H_Cardano(
            K_magnitude(K_R, K_zeta, K_Z, df.q_R),
            inputs.launch_angular_frequency,
            df.epsilon_para,
            df.epsilon_perp,
            df.epsilon_g,
            theta_m,
        )
        if inputs.mode_flag == 1:
            return result[2]
        return result[1]

    def grad_H_Cardano(direction: str, spacing: float):
        return derivative(
            H_cardano,
            direction,
            args={"K_R": df.K_R, "K_zeta": df.K_zeta_initial, "K_Z": df.K_Z},
            spacings=spacing,
        )

    g_R_Cardano = grad_H_Cardano("K_R", inputs.delta_K_R)
    g_zeta_Cardano = grad_H_Cardano("K_zeta", inputs.delta_K_zeta)
    g_Z_Cardano = grad_H_Cardano("K_Z", inputs.delta_K_Z)
    g_magnitude_Cardano = np.sqrt(
        g_R_Cardano**2 + g_zeta_Cardano**2 + g_Z_Cardano**2
    )

    ##
    # From here on, we use the shorthand
    # loc: localisation
    # l_lc: distance from cutoff (l - l_c). Distance along the ray
    # cum: cumulative. As such, cum_loc is the cumulative integral of the localisation
    # p: polarisation
    # r: ray
    # b: beam
    # s: spectrum
    # Otherwise, variable names get really unwieldly
    ##

    # localisation_ray = g_magnitude_Cardano[0]**2/g_magnitude_Cardano**2
    # The first point of the beam may be very slightly in the plasma, so I have used the vacuum expression for the group velocity instead
    loc_r = (
        2 * constants.c / inputs.launch_angular_frequency
    ) ** 2 / g_magnitude_Cardano**2

    # Spectrum piece of localisation as a function of distance along ray
    spectrum_power_law_coefficient = 13 / 3  # Turbulence cascade
    wavenumber_K0 = angular_frequency_to_wavenumber(
        inputs.launch_angular_frequency.data
    )
    loc_s = (k_perp_1_bs / (-2 * wavenumber_K0)) ** (-spectrum_power_law_coefficient)

    # Beam piece of localisation as a function of distance along ray
    # Determinant of the imaginary part of Psi_w
    det_imag_Psi_w_analysis = np.imag(Psi_xx) * np.imag(Psi_yy) - np.imag(Psi_xy) ** 2
    # Determinant of the real part of Psi_w. Not needed for the calculation, but gives useful insight
    det_real_Psi_w_analysis = np.real(Psi_xx) * np.real(Psi_yy) - np.real(Psi_xy) ** 2

    # Assumes circular beam at launch
    beam_waist_y = find_waist(
        inputs.launch_beam_width.data, wavenumber_K0, inputs.launch_beam_curvature.data
    )

    loc_b = (
        (beam_waist_y / np.sqrt(2))
        * det_imag_Psi_w_analysis
        / (np.abs(det_M_w_analysis) * np.sqrt(-np.imag(M_w_inv_yy)))
    )
    # --

    # Polarisation piece of localisation as a function of distance along ray
    # Polarisation e
    # eigenvector corresponding to eigenvalue = 0 (H=0)
    # First, find the components of the tensor D
    # Refer to 21st Dec 2020 notes for more
    # Note that e \cdot e* = 1
    [
        D_11_component,
        D_22_component,
        D_bb_component,
        D_12_component,
        D_1b_component,
    ] = find_D(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df.epsilon_para,
        df.epsilon_perp,
        df.epsilon_g,
        theta_m,
    )

    # Dispersion tensor
    D_tensor = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    D_tensor[:, 0, 0] = D_11_component
    D_tensor[:, 1, 1] = D_22_component
    D_tensor[:, 2, 2] = D_bb_component
    D_tensor[:, 0, 1] = -1j * D_12_component
    D_tensor[:, 1, 0] = 1j * D_12_component
    D_tensor[:, 0, 2] = D_1b_component
    D_tensor[:, 2, 0] = D_1b_component

    H_eigvals, e_eigvecs = np.linalg.eigh(D_tensor)

    # In my experience, H_eigvals[:,1] corresponds to the O mode, and H_eigvals[:,1] corresponds to the X-mode
    # ALERT: This may not always be the case! Check the output figure to make sure that the appropriate solution is indeed 0 along the ray
    # e_hat has components e_1,e_2,e_b
    if inputs.mode_flag == 1:
        H_solver = H_eigvals[:, 1]
        e_hat = e_eigvecs[:, :, 1]
    elif inputs.mode_flag == -1:
        H_solver = H_eigvals[:, 0]
        e_hat = e_eigvecs[:, :, 0]

    # equilibrium dielectric tensor - identity matrix. \bm{\epsilon}_{eq} - \bm{1}
    epsilon_minus_identity = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    identity = np.ones(numberOfDataPoints)
    epsilon_minus_identity[:, 0, 0] = df.epsilon_perp - identity
    epsilon_minus_identity[:, 1, 1] = df.epsilon_perp - identity
    epsilon_minus_identity[:, 2, 2] = df.epsilon_para - identity
    epsilon_minus_identity[:, 0, 1] = -1j * df.epsilon_g
    epsilon_minus_identity[:, 1, 0] = 1j * df.epsilon_g

    # loc_p_unnormalised = abs(contract_special(np.conjugate(e_hat_output), contract_special(epsilon_minus_identity,e_hat_output)))**2 / (electron_density_output*10**19)**2

    # Avoids dividing a small number by another small number, leading to a big number because of numerical errors or something
    loc_p_unnormalised = np.divide(
        np.abs(
            contract_special(
                np.conjugate(e_hat),
                contract_special(epsilon_minus_identity, e_hat),
            )
        )
        ** 2,
        (df.electron_density * 1e19) ** 2,
        out=np.zeros_like(df.electron_density),
        where=df.electron_density > 1e-6,
    )
    loc_p = (
        inputs.launch_angular_frequency**2
        * constants.epsilon_0
        * find_electron_mass(df.get("temperature"))
        / constants.e**2
    ) ** 2 * loc_p_unnormalised
    # Note that loc_p is called varepsilon in my paper

    # Note that K_1 = K cos theta_m, K_2 = 0, K_b = K sin theta_m, as a result of cold plasma dispersion
    K_hat_dot_e_hat = e_hat[:, 0] * np.cos(theta_m) + e_hat[:, 2] * np.sin(theta_m)

    K_hat_dot_e_hat_sq = np.conjugate(K_hat_dot_e_hat) * K_hat_dot_e_hat
    # --

    # TODO: Come back and see if the naming of variables makes sense and is consistent
    # Distance from cutoff
    l_lc = distance_along_line - distance_along_line[cutoff_index]

    # Combining the various localisation pieces to get some overall localisation
    loc_b_r_s = loc_b * loc_r * loc_s
    loc_b_r = loc_b * loc_r

    # Calculates localisation (relevant pieces of the Spherical Tokamak case)
    d_theta_m_d_tau = np.gradient(theta_m, df.tau)
    d_K_d_tau = np.gradient(K_magnitude_array, df.tau)
    # d tau_Booker / d tau_Cardano
    d_tau_B_d_tau_C = g_magnitude_Cardano / df.g_magnitude
    theta_m_min_idx = np.argmin(np.abs(theta_m).data)
    delta_kperp1_ST = k_perp_1_bs - k_perp_1_bs[theta_m_min_idx]
    G_full = (
        (
            d_K_d_tau * df.g_magnitude
            - K_magnitude_array**2 * d_theta_m_d_tau**2 * M_w_inv_xx
        )
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    G_term1 = (d_K_d_tau * df.g_magnitude * d_tau_B_d_tau_C**2) ** (-1)
    G_term2 = (
        K_magnitude_array**2
        * d_theta_m_d_tau**2
        * M_w_inv_xx
        * G_term1**2
        * d_tau_B_d_tau_C**2
    ) ** (-1)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)

    print("Saving analysis data")
    further_df = {
        "Psi_xx": (["tau"], Psi_xx),
        "Psi_xy": (["tau"], Psi_xy),
        "Psi_yy": (["tau"], Psi_yy),
        "Psi_xg": (["tau"], Psi_xg),
        "Psi_yg": (["tau"], Psi_yg),
        "Psi_gg": (["tau"], Psi_gg),
        "Psi_xx_entry": Psi_xx_entry,
        "Psi_xy_entry": Psi_xy_entry,
        "Psi_yy_entry": Psi_yy_entry,
        "Psi_3D_Cartesian": (["tau", "row", "col"], Psi_3D_Cartesian),
        "x_hat_Cartesian": (["tau", "col"], x_hat_Cartesian),
        "y_hat_Cartesian": (["tau", "col"], y_hat_Cartesian),
        "g_hat_Cartesian": (["tau", "col"], g_hat_Cartesian),
        "M_xx": M_xx,
        "M_xy": M_xy,
        "M_yy": M_yy,
        "M_w_inv_xx": M_w_inv_xx,
        "M_w_inv_xy": M_w_inv_xy,
        "M_w_inv_yy": M_w_inv_yy,
        "xhat_dot_grad_bhat_dot_xhat": (["tau"], xhat_dot_grad_bhat_dot_xhat),
        "xhat_dot_grad_bhat_dot_yhat": (["tau"], xhat_dot_grad_bhat_dot_yhat),
        "xhat_dot_grad_bhat_dot_ghat": (["tau"], xhat_dot_grad_bhat_dot_ghat),
        "yhat_dot_grad_bhat_dot_xhat": (["tau"], yhat_dot_grad_bhat_dot_xhat),
        "yhat_dot_grad_bhat_dot_yhat": (["tau"], yhat_dot_grad_bhat_dot_yhat),
        "yhat_dot_grad_bhat_dot_ghat": (["tau"], yhat_dot_grad_bhat_dot_ghat),
        "grad_grad_H": (["tau", "row", "col"], grad_grad_H),
        "gradK_grad_H": (["tau", "row", "col"], gradK_grad_H),
        "gradK_gradK_H": (["tau", "row", "col"], gradK_gradK_H),
        "d_theta_d_tau": (["tau"], d_theta_d_tau),
        "d_xhat_d_tau_dot_yhat": (["tau"], d_xhat_d_tau_dot_yhat),
        "kappa_dot_xhat": (["tau"], kappa_dot_xhat),
        "kappa_dot_yhat": (["tau"], kappa_dot_yhat),
        "kappa_dot_ghat": (["tau"], kappa_dot_ghat),
        "kappa_magnitude": (["tau"], kappa_magnitude),
        "delta_k_perp_2": delta_k_perp_2,
        "delta_theta_m": delta_theta_m,
        "theta_m": theta_m,
        "RZ_distance_along_line": (["tau"], RZ_distance_along_line),
        "distance_along_line": (["tau"], distance_along_line),
        "k_perp_1_bs": k_perp_1_bs,
        "K_magnitude": (["tau"], K_magnitude_array),
        "cutoff_index": cutoff_index,
        "x_hat": df.x_hat,
        "y_hat": df.y_hat,
        "b_hat": df.b_hat,
        "g_hat": df.g_hat,
        "e_hat": (["tau", "col"], e_hat),
        "H_eigvals": (["tau", "col"], H_eigvals),
        "e_eigvecs": (["tau", "row", "col"], e_eigvecs),
        "H_1_Cardano": (["tau"], H_1_Cardano),
        "H_2_Cardano": (["tau"], H_2_Cardano),
        "H_3_Cardano": (["tau"], H_3_Cardano),
        "kperp1_hat": (["tau", "col"], kperp1_hat),
        "theta": (["tau"], theta),
        "g_magnitude_Cardano": g_magnitude_Cardano,
        "poloidal_flux_on_midplane": (["R_midplane"], poloidal_flux_on_midplane),
        "loc_b": loc_b,
        "loc_p": loc_p,
        "loc_r": loc_r,
        "loc_s": loc_s,
        "loc_m": loc_m,
        "loc_b_r_s": loc_b_r_s,
        "loc_b_r": loc_b_r,
    }

    df = df.assign_coords({"R_midplane": R_midplane_points, "l_lc": (["tau"], l_lc)})
    df.update(further_df)

    if detailed_analysis_flag and (cutoff_index + 1 != len(df.tau)):
        detailed_df = detailed_analysis(
            df, cutoff_index, loc_b_r.data, loc_b_r_s.data, l_lc.data, wavenumber_K0
        )
        df.update(detailed_df)

    save_npz(output_path / f"analysis{output_filename_suffix}", df)

    return df


def detailed_analysis(
    df: xr.Dataset,
    cutoff_index: int,
    loc_b_r: FloatArray,
    loc_b_r_s: FloatArray,
    l_lc: FloatArray,
    wavenumber_K0: float,
):
    """
    Now to do some more-complex analysis of the localisation.
    This part of the code fails in some situations, hence I'm making
    it possible to skip this section
    """
    # Finds the 1/e2 values (localisation)
    loc_b_r_s_max_over_e2 = loc_b_r_s.max() / (np.e**2)
    loc_b_r_max_over_e2 = loc_b_r.max() / (np.e**2)

    # Gives the inter-e2 range (analogous to interquartile range) in l-lc
    loc_b_r_s_delta_l_1 = find_x0(
        l_lc[0:cutoff_index], loc_b_r_s[0:cutoff_index], loc_b_r_s_max_over_e2
    )
    loc_b_r_s_delta_l_2 = find_x0(
        l_lc[cutoff_index:], loc_b_r_s[cutoff_index:], loc_b_r_s_max_over_e2
    )
    # The 1/e2 distances,  (l - l_c)
    loc_b_r_s_delta_l = np.array([loc_b_r_s_delta_l_1, loc_b_r_s_delta_l_2])
    loc_b_r_s_half_width_l = (loc_b_r_s_delta_l_2 - loc_b_r_s_delta_l_1) / 2
    loc_b_r_delta_l_1 = find_x0(
        l_lc[0:cutoff_index], loc_b_r[0:cutoff_index], loc_b_r_max_over_e2
    )
    loc_b_r_delta_l_2 = find_x0(
        l_lc[cutoff_index:], loc_b_r[cutoff_index:], loc_b_r_max_over_e2
    )
    # The 1/e2 distances,  (l - l_c)
    loc_b_r_delta_l = np.array([loc_b_r_delta_l_1, loc_b_r_delta_l_2])
    loc_b_r_half_width_l = (loc_b_r_delta_l_1 - loc_b_r_delta_l_2) / 2

    # Estimates the inter-e2 range (analogous to interquartile range) in kperp1, from l-lc
    # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
    loc_b_r_s_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_s_delta_l_1
    )
    loc_b_r_s_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index:], l_lc[cutoff_index:], loc_b_r_s_delta_l_2
    )
    loc_b_r_s_delta_kperp1 = np.array(
        [loc_b_r_s_delta_kperp1_1, loc_b_r_s_delta_kperp1_2]
    )
    loc_b_r_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_delta_l_1
    )
    loc_b_r_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index:], l_lc[cutoff_index:], loc_b_r_delta_l_2
    )
    loc_b_r_delta_kperp1 = np.array([loc_b_r_delta_kperp1_1, loc_b_r_delta_kperp1_2])

    # Calculate the cumulative integral of the localisation pieces
    cum_loc_b_r_s = cumtrapz(loc_b_r_s, df.distance_along_line, initial=0)
    cum_loc_b_r_s = cum_loc_b_r_s - max(cum_loc_b_r_s) / 2
    cum_loc_b_r = cumtrapz(loc_b_r, df.distance_along_line, initial=0)
    cum_loc_b_r = cum_loc_b_r - max(cum_loc_b_r) / 2

    # Finds the 1/e2 values (cumulative integral of localisation)
    cum_loc_b_r_s_max_over_e2 = cum_loc_b_r_s.max() * (1 - 1 / (np.e**2))
    cum_loc_b_r_max_over_e2 = cum_loc_b_r.max() * (1 - 1 / (np.e**2))

    # Gives the inter-e range (analogous to interquartile range) in l-lc
    cum_loc_b_r_s_delta_l_1 = find_x0(l_lc, cum_loc_b_r_s, -cum_loc_b_r_s_max_over_e2)
    cum_loc_b_r_s_delta_l_2 = find_x0(l_lc, cum_loc_b_r_s, cum_loc_b_r_s_max_over_e2)
    cum_loc_b_r_s_delta_l = np.array([cum_loc_b_r_s_delta_l_1, cum_loc_b_r_s_delta_l_2])
    cum_loc_b_r_s_half_width = (cum_loc_b_r_s_delta_l_2 - cum_loc_b_r_s_delta_l_1) / 2
    cum_loc_b_r_delta_l_1 = find_x0(l_lc, cum_loc_b_r, -cum_loc_b_r_max_over_e2)
    cum_loc_b_r_delta_l_2 = find_x0(l_lc, cum_loc_b_r, cum_loc_b_r_max_over_e2)
    cum_loc_b_r_delta_l = np.array([cum_loc_b_r_delta_l_1, cum_loc_b_r_delta_l_2])
    cum_loc_b_r_half_width = (cum_loc_b_r_delta_l_2 - cum_loc_b_r_delta_l_1) / 2

    # Gives the inter-e2 range (analogous to interquartile range) in kperp1.
    # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
    cum_loc_b_r_s_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index],
        cum_loc_b_r_s[0:cutoff_index],
        -cum_loc_b_r_s_max_over_e2,
    )
    cum_loc_b_r_s_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index::],
        cum_loc_b_r_s[cutoff_index::],
        cum_loc_b_r_s_max_over_e2,
    )
    cum_loc_b_r_s_delta_kperp1 = np.array(
        [cum_loc_b_r_s_delta_kperp1_1, cum_loc_b_r_s_delta_kperp1_2]
    )
    cum_loc_b_r_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index],
        cum_loc_b_r[0:cutoff_index],
        -cum_loc_b_r_max_over_e2,
    )
    cum_loc_b_r_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index::],
        cum_loc_b_r[cutoff_index::],
        cum_loc_b_r_max_over_e2,
    )
    cum_loc_b_r_delta_kperp1 = np.array(
        [cum_loc_b_r_delta_kperp1_1, cum_loc_b_r_delta_kperp1_2]
    )

    # Gives the mode l-lc for backscattering
    loc_b_r_s_max_index = find_nearest(loc_b_r_s, loc_b_r_s.max())
    loc_b_r_s_max_l_lc = (
        df.distance_along_line[loc_b_r_s_max_index]
        - df.distance_along_line[cutoff_index]
    )
    loc_b_r_max_index = find_nearest(loc_b_r, loc_b_r.max())
    loc_b_r_max_l_lc = (
        df.distance_along_line[loc_b_r_max_index] - df.distance_along_line[cutoff_index]
    )

    # Gives the mean l-lc for backscattering
    cum_loc_b_r_s_mean_l_lc = (
        np.trapz(loc_b_r_s * df.distance_along_line, df.distance_along_line)
        / np.trapz(loc_b_r_s, df.distance_along_line)
        - df.distance_along_line[cutoff_index]
    )
    cum_loc_b_r_mean_l_lc = (
        np.trapz(loc_b_r * df.distance_along_line, df.distance_along_line)
        / np.trapz(loc_b_r, df.distance_along_line)
        - df.distance_along_line[cutoff_index]
    )

    # Gives the median l-lc for backscattering
    cum_loc_b_r_s_delta_l_0 = find_x0(l_lc, cum_loc_b_r_s, 0)
    cum_loc_b_r_delta_l_0 = find_x0(l_lc, cum_loc_b_r, 0)

    # Due to the divergency of the ray piece, the mode kperp1 for backscattering is exactly that at the cut-off

    # Gives the mean kperp1 for backscattering
    cum_loc_b_r_s_mean_kperp1 = np.trapz(
        loc_b_r_s * df.k_perp_1_bs, df.k_perp_1_bs
    ) / np.trapz(loc_b_r_s, df.k_perp_1_bs)
    cum_loc_b_r_mean_kperp1 = np.trapz(
        loc_b_r * df.k_perp_1_bs, df.k_perp_1_bs
    ) / np.trapz(loc_b_r, df.k_perp_1_bs)

    # Gives the median kperp1 for backscattering
    cum_loc_b_r_s_delta_kperp1_0 = find_x0(df.k_perp_1_bs, cum_loc_b_r_s, 0)
    # Only works if point is before cutoff. To fix.
    cum_loc_b_r_delta_kperp1_0 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], cum_loc_b_r[0:cutoff_index], 0
    )

    # To make the plots look nice
    k_perp_1_bs_plot = np.append(-2 * wavenumber_K0, df.k_perp_1_bs)
    k_perp_1_bs_plot = np.append(k_perp_1_bs_plot, -2 * wavenumber_K0)
    cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s[0], cum_loc_b_r_s)
    cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s_plot, cum_loc_b_r_s[-1])
    cum_loc_b_r_plot = np.append(cum_loc_b_r[0], cum_loc_b_r)
    cum_loc_b_r_plot = np.append(cum_loc_b_r_plot, cum_loc_b_r[-1])

    # These will get added as "dimension coordinates", arrays with
    # coordinates that have the same name, because we've not specified
    # dimensions, which is perhaps not desired. What might be better
    # would be to use e.g. loc_b_r_delta_l as the coordinate for
    # loc_b_r_max_over_e2
    return {
        "loc_b_r_s_max_over_e2": loc_b_r_s_max_over_e2,
        "loc_b_r_max_over_e2": loc_b_r_max_over_e2,
        # The 1/e2 distances,  (l - l_c)
        "loc_b_r_s_delta_l": loc_b_r_s_delta_l,
        "loc_b_r_delta_l": loc_b_r_delta_l,
        # The 1/e2 distances, kperp1, estimated from (l - l_c)
        "loc_b_r_s_delta_kperp1": loc_b_r_s_delta_kperp1,
        "loc_b_r_delta_kperp1": loc_b_r_delta_kperp1,
        "cum_loc_b_r_s": cum_loc_b_r_s,
        "cum_loc_b_r": cum_loc_b_r,
        "k_perp_1_bs_plot": k_perp_1_bs_plot,
        "cum_loc_b_r_s_plot": cum_loc_b_r_s_plot,
        "cum_loc_b_r_plot": cum_loc_b_r_plot,
        "cum_loc_b_r_s_max_over_e2": cum_loc_b_r_s_max_over_e2,
        "cum_loc_b_r_max_over_e2": cum_loc_b_r_max_over_e2,
        "cum_loc_b_r_s_delta_l": cum_loc_b_r_s_delta_l,
        "cum_loc_b_r_delta_l": cum_loc_b_r_delta_l,
        "cum_loc_b_r_s_delta_kperp1": cum_loc_b_r_s_delta_kperp1,
        "cum_loc_b_r_delta_kperp1": cum_loc_b_r_delta_kperp1,
        "loc_b_r_s_max_l_lc": loc_b_r_s_max_l_lc,
        "loc_b_r_max_l_lc": loc_b_r_max_l_lc,
        "cum_loc_b_r_s_mean_l_lc": cum_loc_b_r_s_mean_l_lc,
        "cum_loc_b_r_mean_l_lc": cum_loc_b_r_mean_l_lc,  # mean l-lc
        "cum_loc_b_r_s_delta_l_0": cum_loc_b_r_s_delta_l_0,
        "cum_loc_b_r_delta_l_0": cum_loc_b_r_delta_l_0,  # median l-lc
        "cum_loc_b_r_s_mean_kperp1": cum_loc_b_r_s_mean_kperp1,
        "cum_loc_b_r_mean_kperp1": cum_loc_b_r_mean_kperp1,  # mean kperp1
        "cum_loc_b_r_s_delta_kperp1_0": cum_loc_b_r_s_delta_kperp1_0,
    }
