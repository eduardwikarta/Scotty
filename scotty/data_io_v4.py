import logging
from scotty.checks_v4 import Parameters, VALID_FIELDS
import xarray as xr

log = logging.getLogger(__name__)

def inputs_to_dataset(params: Parameters, field: VALID_FIELDS) -> xr.Dataset:

    cart = params.cartesian_flag

    inputs_ds = xr.Dataset({
        # Main parameters
        "geometry": params.geometry,
        "poloidal_launch_angle_Torbeam": params.poloidal_launch_angle_deg_Torbeam,
        "toroidal_launch_angle_Torbeam": params.toroidal_launch_angle_deg_Torbeam,
        "launch_freq_GHz": params.launch_frequency_GHz,
        # "launch_angular_frequency": params.launch_angular_frequency,
        # "launch_wavenumber": params.launch_wavenumber,
        "launch_beam_width": params.launch_beam_width,
        "launch_beam_curvature": params.launch_beam_curvature,
        "launch_position": params.q_launch,
        "mode_flag": params.mode_flag_launch,

        # Solver settings and finite-difference parameters
        "ray_tracing_flag": params.ray_tracing_flag,
        "launch_flag": params.launch_flag,
        "boundary_flag": params.boundary_flag,
        "relativistic_flag": params.relativistic_flag,
        "auto_delta_sign": params.auto_delta_sign,
        **({"delta_X": params.delta_X} if cart else {"delta_R": params.delta_R}),
        **({"delta_Y": params.delta_Y} if cart else {}),
        "delta_Z": params.delta_Z,
        **({"delta_K_X": params.delta_K_X} if cart else {"delta_K_R": params.delta_K_R}),
        **({"delta_K_Y": params.delta_K_Y} if cart else {"delta_K_zeta": params.delta_K_zeta}),
        "delta_K_Z": params.delta_K_Z,
        "len_tau": params.len_tau,
        "rtol": params.rtol,
        "atol": params.atol,
        "poloidal_flux_enter": params.poloidal_flux_enter,
        "poloidal_flux_zero_density": params.poloidal_flux_zero_density,
        "poloidal_flux_zero_temperature": params.poloidal_flux_zero_temperature,

        # Data input and output arguments
        "find_B_method": params.find_B_method,
        "magnetic_data_path": params.magnetic_data_path,
        "magnetic_data_interp_order": params.interp_order_magnetic_data_int,

        "ne_data_path": params.ne_data_path,
        "ne_data_interp_order": params.interp_order_ne_data, # int
        "density_fit_parameters": params.density_fit_parameters,
        "density_fit_method": params.density_fit_method,

        "Te_data_path": params.Te_data_path,
        "Te_data_interp_order": params.interp_order_Te_data,
        "temperature_fit_parameters": params.temperature_fit_parameters,
        "temperature_fit_method": params.temperature_fit_method,

        "interp_smoothing": params.interp_smoothing,
        "shot": params.shot,
        "equil_time": params.equil_time,
        
        "input_filename_suffix": params.input_filename_suffix,
        "output_path": params.output_path,
        "output_filename_suffix": params.output_filename_suffix,

        # Analysis and plotting arguments
        "figure_flag": params.figure_flag,
        "further_analysis_flag": params.further_analysis_flag,
        "detailed_analysis_flag": params.detailed_analysis_flag,
        },
        coords = {
            **({"X": field.X_coord} if cart else {"R": field.R_coord}), # type: ignore
            **({"Y": field.Y_coord} if cart else {}), # type: ignore
            "Z": field.Z_coord,
            "row": ["X","Y","Z"] if cart else ["R","zeta","Z"],
            "col": ["X","Y","Z"] if cart else ["R","zeta","Z"],
        },
    )

    return inputs_ds

def solver_to_dataset(params: Parameters, field: VALID_FIELDS) -> xr.Dataset:

    cart = params.cartesian_flag
    ray_tracing_flag = params.ray_tracing_flag

    solver_ds = xr.Dataset({
            "solver_status": params.solver_status,
            "tau_array": params.tau_output,
            "a": 1,
            }.update({} if ray_tracing_flag else {
            "": 1,
        }),
        coords = {
            "a": 1,
        },
    )