import logging
import numpy as np
from scipy import constants
# from scotty.checks_v4 import VALID_LAUNCH_MODE_FLAGS
VALID_LAUNCH_MODE_FLAGS = 1
from scotty.logger_v4 import arr2str
from scotty.typing import ArrayLike, Array, FloatArray, ComplexFloatArray
from typing import Callable, Union, Tuple, Optional, List

log = logging.getLogger(__name__)

##################################################
#
# UNIT CONVERSIONS
#
##################################################

def freq_GHz_to_angular_frequency(freq_GHz: float) -> float:
    """Convert frequency in GHz to angular frequency"""
    return 2 * np.pi * 1e9 * freq_GHz

def angular_frequency_to_wavenumber(angular_frequency: float) -> float:
    """Convert angular frequency to wavenumber"""
    return angular_frequency / constants.c

def freq_GHz_to_wavenumber(freq_GHz: float) -> float:
    """Converts frequency in GHz to wavenumber"""
    return angular_frequency_to_wavenumber(freq_GHz_to_angular_frequency(freq_GHz))

##################################################
#
# COORDINATE TRANSFORMS
#
##################################################

def find_q_labframe_cart_to_cyl(q_lab_cart: FloatArray) -> FloatArray:
    """
    Converts q_labframe from Cartesian to cylindrical coordinates, both in the lab frame (not the beam frame)
    The shape of q_lab_cart must be either (3,) or (3, N), where N is the number of points at which q is evaluated
    """
    if (q_lab_cart.ndim == 1 or q_lab_cart.ndim == 2) and q_lab_cart.shape[0] == 3: q_X, q_Y, q_Z = q_lab_cart
    else: raise ValueError(f"`find_q_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cart.shape}")
    return np.array([np.sqrt(q_X**2 + q_Y**2), np.arctan2(q_Y, q_X), q_Z])

def find_q_labframe_cyl_to_cart(q_lab_cyl: FloatArray) -> FloatArray:
    """
    Converts q_labframe from cylindrical to Cartesian coordinates, both in the lab frame (not the beam frame)
    The shape of q_lab_cyl must be either (3,) or (3, N), where N is the number of points at which q is evaluated
    """
    if (q_lab_cyl.ndim == 1 or q_lab_cyl.ndim == 2) and q_lab_cyl.shape[0] == 3: q_R, q_p, q_Z = q_lab_cyl
    else: raise ValueError(f"`find_q_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cyl.shape}")
    return np.array([q_R*np.cos(q_p), q_R*np.sin(q_p), q_Z])

def find_K_labframe_cart_to_cyl(K_lab_cart: FloatArray, q_lab_cart: FloatArray) -> FloatArray:
    """
    Converts K_labframe from Cartesian to cylindrical coordinates, both in the lab frame (not the beam frame)
    The shape of K_lab_cart must be either (3,) or (3, N), where N is the number of points at which K is evaluated,
    and the shapes of K_lab_cart and q_lab_cart must be the same
    """
    if (q_lab_cart.ndim == 1 or q_lab_cart.ndim == 2) and q_lab_cart.shape[0] == 3: q_R, q_zeta, q_Z = find_q_labframe_cart_to_cyl(q_lab_cart = q_lab_cart)
    else: raise ValueError(f"`find_K_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cart.shape}")

    if (K_lab_cart.ndim == 1 or K_lab_cart.ndim == 2) and K_lab_cart.shape[0] == 3: K_X, K_Y, K_Z = K_lab_cart
    else: raise ValueError(f"`find_K_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {K_lab_cart.shape}")

    if not K_lab_cart.shape == q_lab_cart.shape: raise ValueError(f"`find_K_labframe_cart_to_cyl`: Expected `K_lab_cart` and `q_lab_cart` to have the same shape, but got {K_lab_cart.shape} and {q_lab_cart.shape}")

    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    return np.array([
        K_X * cos_zeta + K_Y * sin_zeta,
        (-K_X * sin_zeta + K_Y * cos_zeta) * q_R,
        K_Z
    ])

def find_K_labframe_cyl_to_cart(K_lab_cyl: FloatArray, q_lab_cyl: FloatArray) -> FloatArray:
    """
    Converts K_labframe from cylindrical to Cartesian coordinates, both in the lab frame (not the beam frame)
    The shape of K_lab_cyl must be either (3,) or (3, N), where N is the number of points at which K is evaluated,
    and the shapes of K_lab_cyl and q_lab_cyl must be the same
    """
    if (q_lab_cyl.ndim == 1 or q_lab_cyl.ndim == 2) and q_lab_cyl.shape[0] == 3: q_R, q_zeta, q_Z = q_lab_cyl
    else: raise ValueError(f"`find_K_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cyl.shape}")

    if (K_lab_cyl.ndim == 1 or K_lab_cyl.ndim == 2) and K_lab_cyl.shape[0] == 3: K_R, K_zeta, K_Z = K_lab_cyl
    else: raise ValueError(f"`find_K_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {K_lab_cyl.shape}")

    if not K_lab_cyl.shape == q_lab_cyl.shape: raise ValueError(f"`find_K_labframe_cyl_to_cart`: Expected `K_lab_cyl` and `q_lab_cyl` to have the same shape, but got {K_lab_cyl.shape} and {q_lab_cyl.shape}")
    
    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    return np.array([
        K_R * cos_zeta - K_zeta * sin_zeta / q_R,
        K_R * sin_zeta + K_zeta * cos_zeta / q_R,
        K_Z
    ])

def find_Psi_3D_labframe_cart_to_cyl(Psi_3D_labframe_cartesian: ComplexFloatArray, K_lab_cart: FloatArray, q_lab_cart: FloatArray) -> ComplexFloatArray:
    """
    Converts Psi_3D_labframe from Cartesian to cylindrical coordinates, both in the lab frame (not the beam frame)
    The shape of Psi_3D_labframe_cartesian must be either (3, 3) or (N, 3, 3), where N is the number of points at which Psi is evaluated,
    with the shapes of K_lab_cart and q_lab_cart being the same size and also compatible with Psi_3D_labframe_cartesian (i.e. if
    Psi_3D_labframe_cartesian is (3, 3) then K_lab_cart and q_lab_cart must be (3,), and if Psi_3D_labframe_cartesian is (N, 3, 3)
    then K_lab_cart and q_lab_cart must be (3, N)). Note that the shape of Psi_3D_labframe_cartesian is different from the shapes
    of K_lab_cart and q_lab_cart because of the way we evolve the beam
    """
    q_R, q_zeta, q_Z = find_q_labframe_cart_to_cyl(q_lab_cart)
    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)
    K_R, K_zeta, K_Z = find_K_labframe_cart_to_cyl(K_lab_cart, q_lab_cart)

    if Psi_3D_labframe_cartesian.ndim == 2: # A single matrix of Psi
        Psi = np.array([Psi_3D_labframe_cartesian])
        N = 1
        squeeze = True
    elif Psi_3D_labframe_cartesian.ndim == 3: # A stack of matrices of Psi
        Psi = Psi_3D_labframe_cartesian
        N = Psi.shape[0]
        squeeze = False
    else: raise ValueError(f"`Psi_3D_labframe_cartesian` has an invalid number of dimensions: Expected 2 or 3 but got ndim = {Psi_3D_labframe_cartesian.ndim}")

    Psi_XX = Psi[:, 0, 0]
    Psi_XY = Psi[:, 0, 1]
    Psi_XZ = Psi[:, 0, 2]
    Psi_YY = Psi[:, 1, 1]
    Psi_YZ = Psi[:, 1, 2]
    Psi_ZZ = Psi[:, 2, 2]

    Psi_cyl = np.zeros([N,3,3], dtype="complex128")
    Psi_cyl[:, 0, 0] = Psi_XX * cos_zeta**2 + 2 * Psi_XY * sin_zeta * cos_zeta + Psi_YY * sin_zeta**2
    Psi_cyl[:, 0, 1] = (-Psi_XX * sin_zeta * cos_zeta + Psi_XY * (cos_zeta**2 - sin_zeta**2) + Psi_YY * sin_zeta * cos_zeta) * q_R + K_zeta / q_R
    Psi_cyl[:, 0, 2] = Psi_XZ * cos_zeta + Psi_YZ * sin_zeta
    Psi_cyl[:, 1, 1] = (Psi_XX * sin_zeta**2 - 2 * Psi_XY * sin_zeta * cos_zeta + Psi_YY * cos_zeta**2) * q_R**2 - K_R * q_R
    Psi_cyl[:, 1, 2] = (-Psi_XZ * sin_zeta + Psi_YZ * cos_zeta) * q_R
    Psi_cyl[:, 2, 2] = Psi_ZZ
    Psi_cyl[:, 1, 0] = Psi_cyl[:, 0, 1]
    Psi_cyl[:, 2, 0] = Psi_cyl[:, 0, 2]
    Psi_cyl[:, 2, 1] = Psi_cyl[:, 1, 2]

    return Psi_cyl[0] if squeeze else Psi_cyl

def find_Psi_3D_labframe_cyl_to_cart(Psi_3D_labframe_cylindrical: ComplexFloatArray, K_lab_cyl: FloatArray, q_lab_cyl: FloatArray) -> ComplexFloatArray:
    """
    Converts Psi_3D_labframe from cylindrical to Cartesian coordinates, both in the lab frame (not the beam frame)
    The shape of Psi_3D_labframe_cylindrical must be either (3, 3) or (N, 3, 3), where N is the number of points at which Psi is evaluated,
    with the shapes of K_lab_cyl and q_lab_cyl being the same size and also compatible with Psi_3D_labframe_cylindrical (i.e. if
    Psi_3D_labframe_cylindrical is (3, 3) then K_lab_cyl and q_lab_cyl must be (3,), and if Psi_3D_labframe_cylindrical is (N, 3, 3)
    then K_lab_cyl and q_lab_cyl must be (3, N)). Note that the shape of Psi_3D_labframe_cylindrical is different from the shapes
    of K_lab_cyl and q_lab_cyl because of the way we evolve the beam
    """
    q_R, q_zeta, q_Z = q_lab_cyl
    q_X, q_Y, _ = find_q_labframe_cyl_to_cart(q_lab_cyl)
    K_R, K_zeta, K_Z = K_lab_cyl

    if Psi_3D_labframe_cylindrical.ndim == 2: # A single matrix of Psi
        Psi = np.array([Psi_3D_labframe_cylindrical])
        N = 1
        squeeze = True
    elif Psi_3D_labframe_cylindrical.ndim == 3: # A stack of matrices of Psi
        Psi = Psi_3D_labframe_cylindrical
        N = Psi.shape[0]
        squeeze = False
    else: raise ValueError(f"`Psi_3D_labframe_cylindrical` has an invalid number of dimensions: Expected 2 or 3 but got ndim = {Psi_3D_labframe_cylindrical.ndim}")

    Psi_RR       = Psi[:, 0, 0]
    Psi_Rzeta    = Psi[:, 0, 1]
    Psi_RZ       = Psi[:, 0, 2]
    Psi_zetazeta = Psi[:, 1, 1]
    Psi_zetaZ    = Psi[:, 1, 2]
    Psi_ZZ       = Psi[:, 2, 2]

    Psi_temp = np.zeros([N,3,3], dtype="complex128")
    Psi_temp[:, 0, 0] = Psi_RR
    Psi_temp[:, 0, 1] = Psi_Rzeta / q_R - K_zeta / q_R**2
    Psi_temp[:, 0, 2] = Psi_RZ
    Psi_temp[:, 1, 1] = Psi_zetazeta / q_R**2 + K_R / q_R
    Psi_temp[:, 1, 2] = Psi_zetaZ / q_R
    Psi_temp[:, 2, 2] = Psi_ZZ
    Psi_temp[:, 1, 0] = Psi_temp[:, 0, 1]
    Psi_temp[:, 2, 0] = Psi_temp[:, 0, 2]
    Psi_temp[:, 2, 1] = Psi_temp[:, 1, 2]

    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)
    zeros = np.zeros_like(q_zeta)
    ones = np.ones_like(q_zeta)

    rotation_matrix_xi = np.moveaxis(np.array([
        [cos_zeta, -sin_zeta, zeros],
        [sin_zeta,  cos_zeta, zeros],
        [zeros,     zeros,    ones ],
    ]), -1, 0) # moveaxis to for broadcasting as (N,3,3)

    rotation_matrix_xi_inverse = np.moveaxis(np.swapaxes(rotation_matrix_xi, 0, 1), -1, 0) # same moveaxis as above

    Psi_cart = np.matmul(np.matmul(rotation_matrix_xi_inverse, Psi_temp), rotation_matrix_xi)

    return Psi_cart[0] if squeeze else Psi_cart

def find_vector_and_q_cyl_to_cart(vector_labframe_cyl: FloatArray, q_labframe_cyl: FloatArray) -> Tuple[FloatArray, FloatArray]:
    if not vector_labframe_cyl.shape == q_labframe_cyl.shape: raise ValueError(f"`vector_labframe_cyl` and `q_labframe_cyl` must have the same shape, but got {vector_labframe_cyl.shape} and {q_labframe_cyl.shape}")
    elif not vector_labframe_cyl.shape[0] == 3: raise ValueError(f"`vector_labframe_cyl` must be indexed as a 3-vector")
    elif not q_labframe_cyl.shape[0] == 3: raise ValueError(f"`q_labframe_cyl` must be indexed as a 3-vector")

    v_R, v_zeta, v_Z = vector_labframe_cyl
    q_R, q_zeta, q_Z = q_labframe_cyl
    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    q_labframe_cart = np.zeros_like(q_labframe_cyl, dtype=q_labframe_cyl.dtype)
    q_labframe_cart[0] = q_R*cos_zeta
    q_labframe_cart[1] = q_R*sin_zeta
    q_labframe_cart[2] = q_Z

    v_labframe_cart = np.zeros_like(vector_labframe_cyl, dtype=vector_labframe_cyl.dtype)
    v_labframe_cart[0] = v_R*cos_zeta - v_zeta*sin_zeta
    v_labframe_cart[1] = v_R*sin_zeta + v_zeta*cos_zeta
    v_labframe_cart[2] = v_Z

    return v_labframe_cart, q_labframe_cart

##################################################
#
# GENERAL FUNCTIONS
#
##################################################

def ray_line(
    q_X_launch: float, q_Y_launch: float, q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
) -> FloatArray:
    
    """Returns `q_X`, `q_Y`, `q_Z`"""

    XYZ_start = np.array([q_X_launch, q_Y_launch, q_Z_launch])

    # This parametrises the ray in a line away from the antenna,
    # at a/up to some values of the parameter `tau`
    poloidal_launch_angle = -np.deg2rad(poloidal_launch_angle_deg_Torbeam)
    toroidal_launch_angle =  np.deg2rad(toroidal_launch_angle_deg_Torbeam) + np.pi
    XYZ_step = np.array([np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
                         np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
                         np.sin(poloidal_launch_angle)])

    ray_line_positions = XYZ_start + np.outer(tau, XYZ_step)

    return np.squeeze(ray_line_positions).T

def poloidal_flux_along_ray_line(
    q_X_launch: float, q_Y_launch: float, q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    _field_cart_polflux: Callable):
    
    positions = ray_line(q_X_launch, q_Y_launch, q_Z_launch, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)

    # Get the poloidal flux values at a particular point or point(s).
    # If there are NaNs, then replace those with the first non-NaN instance
    # in the array (i.e. the poloidal flux of the first point in the field)
    if positions.ndim == 1:
        polflux = _field_cart_polflux(*positions)
        if np.isnan(polflux):
            log.warning(f"Poloidal flux is NaN at [X,Y,Z] = [{arr2str(positions)}]. Returning None instead")
            polflux = np.nan
    
    elif positions.ndim == 2:
        polflux = _field_cart_polflux(positions[:, 0], positions[:, 1], positions[:, 2])
        if np.all(np.isnan(polflux)):
            _printmsg = "\n".join(f"            - [{position}]" for position in positions)
            log.warning(f"""
        Poloidal fluxes are NaNs at all points queried: [X,Y,Z] = \n{_printmsg}
        """)
            polflux = np.nan
        else:
            NaN_replacement_value = polflux[~np.isnan(polflux)][0]
            polflux = np.nan_to_num(polflux, nan=NaN_replacement_value)
    
    return polflux # type: ignore

def poloidal_flux_difference_along_ray_line(
    q_X_launch: float, q_Y_launch: float, q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    _field_cart_polflux: Callable,
    poloidal_flux_enter: float):
    
    polflux = poloidal_flux_along_ray_line(q_X_launch, q_Y_launch, q_Z_launch, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, _field_cart_polflux)

    # If NaN, then just return NaN
    if polflux is np.nan: return polflux
    else: return polflux - poloidal_flux_enter

# TO REMOVE -- rewrite using Einstein summation convention?
def dot(a: Array, b: Array) -> Array:
    """Dot product of arrays of vectors or matrices.

    Covers the case that matmul and dot don't do very elegantly, and
    avoids having to use a for loop to iterate over the array slices.

    Replaces the deprecated contract_special.

    Parameters
    ----------
    arg_a, arg_b:
        One of:

        - matrix with shape TxMxN and a vector with TxN or TxM
        - two vectors of shape TxN

        For each T, independently compute the dot product of the
        matrices/vectors

    """
    return np.array(list(map(np.dot, a, b)))

def make_array_3x3(arr: Array) -> Array:
    r"""Convert a 2x2 array into a 3x3 by appending zeros on the outside:

    .. math::

        \begin{pmatrix}
            a & b \\
            c & d \\
        \end{pmatrix}
        \Rightarrow
        \begin{pmatrix}
            a & b & 0 \\
            c & d & 0 \\
            0 & 0 & 0 \\
        \end{pmatrix}
    """
    if arr.shape != (2, 2): raise ValueError(f"Expected array shape to be (2, 2), got {arr.shape}")
    out = np.zeros((3, 3), dtype=arr.dtype)
    out[:2, :2] = arr
    return out

def find_inverse_2D(matrix_2D: Array) -> Array:
    # Finds the inverse of a 2x2 matrix
    matrix_2D_inverse = np.zeros([2, 2], dtype=matrix_2D.dtype)
    determinant = matrix_2D[0, 0] * matrix_2D[1, 1] - matrix_2D[0, 1] * matrix_2D[1, 0]
    matrix_2D_inverse[0, 0] =  matrix_2D[1, 1] / determinant
    matrix_2D_inverse[1, 1] =  matrix_2D[0, 0] / determinant
    matrix_2D_inverse[0, 1] = -matrix_2D[0, 1] / determinant
    matrix_2D_inverse[1, 0] = -matrix_2D[1, 0] / determinant
    return matrix_2D_inverse

def find_K_magnitude(cartesian: bool, K0: ArrayLike, K1: ArrayLike, K2: ArrayLike, q0: ArrayLike):
    return np.sqrt(K0**2 + K1**2 + K2**2) if cartesian else np.sqrt(K0**2 + (K1/q0)**2 + K2**2)

##################################################
#
# PLASMA CALCULATIONS
#
##################################################

def find_electron_mass(temperature: Optional[ArrayLike] = None) -> ArrayLike:
    # Mazzucato's relativistic correction
    factor = 1 if temperature is None else 1 + temperature * 4.892 * 10**(-3)
    return factor * constants.m_e

def find_normalised_plasma_freq(launch_angular_freq: float, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    m_e = find_electron_mass(temperature)
    return (constants.e * np.sqrt(electron_density * 10**19 / (constants.epsilon_0 * m_e))) / launch_angular_freq

def find_normalised_gyro_freq(launch_angular_freq: float, B_total: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    m_e = find_electron_mass(temperature)
    return constants.e * B_total / (m_e * launch_angular_freq)

def find_normalised_cutoff_and_hybrid_freqs(launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> List[ArrayLike]:
    r"""Returns `(normalised_lefthand_cutoff, normalised_righthand_cutoff, normalised_upper_hybrid)`"""
    normalised_plasma_freq = find_normalised_plasma_freq(launch_angular_freq, electron_density, temperature)
    normalised_gyro_freq = find_normalised_gyro_freq(launch_angular_freq, B_total, temperature)
    normalised_lefthand_cutoff  = 0.5 * (-normalised_gyro_freq + np.sqrt(normalised_gyro_freq**2 + 4 * normalised_plasma_freq**2))
    normalised_righthand_cutoff = 0.5 * ( normalised_gyro_freq + np.sqrt(normalised_gyro_freq**2 + 4 * normalised_plasma_freq**2))
    normalised_upper_hybrid = np.sqrt(normalised_plasma_freq**2 + normalised_gyro_freq**2)
    return [normalised_lefthand_cutoff, normalised_righthand_cutoff, normalised_upper_hybrid]

def find_epsilon_terms(
    launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None,
    para: bool = True, perp: bool = True, g: bool = True
) -> List[ArrayLike]:
    r"""Returns `[epsilon_para, epsilon_perp, epsilon_g]` if all flags are `True`.
    Otherwise, the corresponding term is omitted but everything else is ordered the same"""
    toreturn = []
    normalised_plasma_freq = find_normalised_plasma_freq(launch_angular_freq, electron_density, temperature)
    normalised_gyro_freq = find_normalised_gyro_freq(launch_angular_freq, B_total, temperature)

    if para: toreturn.append(1 - normalised_plasma_freq**2)
    if perp: toreturn.append(1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2))
    if g:    toreturn.append(normalised_plasma_freq**2 * normalised_gyro_freq / (1 - normalised_gyro_freq**2))
    return toreturn

def find_epsilon_para(launch_angular_freq: float, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `epsilon_para`. Also called epsilon_bb in the paper"""
    return find_epsilon_terms(launch_angular_freq, 0, electron_density, temperature, para=True, perp=False, g=False)[0]

def find_epsilon_perp(launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `epsilon_perp`. Also called epsilon_11 in the paper"""
    return find_epsilon_terms(launch_angular_freq, B_total, electron_density, temperature, para=False, perp=True, g=False)[0]

def find_epsilon_g(launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `epsilon_g`. Also called epsilon_12 in the paper"""
    return find_epsilon_terms(launch_angular_freq, B_total, electron_density, temperature, para=False, perp=False, g=True)[0]

# def find_epsilon_para(launch_angular_freq: float, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     r"""Also called epsilon_bb in the paper"""
#     normalised_plasma_freq = find_normalised_plasma_freq(launch_angular_freq, electron_density, temperature)
#     return 1 - normalised_plasma_freq**2

# def find_epsilon_perp(launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     r"""Also called epsilon_11 in the paper"""
#     normalised_plasma_freq = find_normalised_plasma_freq(launch_angular_freq, electron_density, temperature)
#     normalised_gyro_freq = find_normalised_gyro_freq(launch_angular_freq, B_total, temperature)
#     return 1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2)

# def find_epsilon_g(launch_angular_freq: float, B_total: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     r"""Also called epsilon_12 in the paper"""
#     normalised_plasma_freq = find_normalised_plasma_freq(launch_angular_freq, electron_density, temperature)
#     normalised_gyro_freq = find_normalised_gyro_freq(launch_angular_freq, B_total, temperature)
#     return (normalised_plasma_freq**2 * normalised_gyro_freq / (1 - normalised_gyro_freq**2))

def find_Booker_terms(
    launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None,
    alpha: bool = True, beta: bool = True, gamma: bool = True
) -> List[ArrayLike]:
    r"""Returns `[Booker_alpha, Booker_beta, Booker_gamma]` if all flags are `True`.
    Otherwise, the corresponding term is omitted but everything else is ordered the same"""
    toreturn = []
    epsilon_para, epsilon_perp, epsilon_g = find_epsilon_terms(launch_angular_freq, B_total, electron_density, temperature)

    if alpha: toreturn.append(epsilon_para * sin_theta_m_sq + epsilon_perp * (1 - sin_theta_m_sq))
    if beta:  toreturn.append(-epsilon_perp * epsilon_para * (1 + sin_theta_m_sq) - (epsilon_perp**2 - epsilon_g**2) * (1 - sin_theta_m_sq))
    if gamma: toreturn.append(epsilon_para * (epsilon_perp**2 - epsilon_g**2))
    return toreturn

def find_Booker_alpha(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `Booker_alpha`"""
    return find_Booker_terms(launch_angular_freq, B_total, sin_theta_m_sq, electron_density, temperature, alpha=True, beta=False, gamma=False)[0]

def find_Booker_beta(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `Booker_beta`"""
    return find_Booker_terms(launch_angular_freq, B_total, sin_theta_m_sq, electron_density, temperature, alpha=False, beta=True, gamma=False)[0]

def find_Booker_gamma(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
    r"""Returns `Booker_gamma`"""
    return find_Booker_terms(launch_angular_freq, B_total, sin_theta_m_sq, electron_density, temperature, alpha=False, beta=False, gamma=True)[0]

# def find_Booker_alpha(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     epsilon_para = find_epsilon_para(launch_angular_freq, electron_density, temperature)
#     epsilon_perp = find_epsilon_perp(launch_angular_freq, B_total, electron_density, temperature)
#     return epsilon_para * sin_theta_m_sq + epsilon_perp * (1 - sin_theta_m_sq)

# def find_Booker_beta(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     epsilon_para = find_epsilon_para(launch_angular_freq, electron_density, temperature)
#     epsilon_perp = find_epsilon_perp(launch_angular_freq, B_total, electron_density, temperature)
#     epsilon_g = find_epsilon_g(launch_angular_freq, B_total, electron_density, temperature)
#     return -epsilon_perp * epsilon_para * (1 + sin_theta_m_sq) - (epsilon_perp**2 - epsilon_g**2) * (1 - sin_theta_m_sq)

# def find_Booker_gamma(launch_angular_freq: float, B_total: ArrayLike, sin_theta_m_sq: ArrayLike, electron_density: ArrayLike, temperature: Optional[ArrayLike] = None) -> ArrayLike:
#     epsilon_para = find_epsilon_para(launch_angular_freq, electron_density, temperature)
#     epsilon_perp = find_epsilon_perp(launch_angular_freq, B_total, electron_density, temperature)
#     epsilon_g = find_epsilon_g(launch_angular_freq, B_total, electron_density, temperature)
#     return epsilon_para * (epsilon_perp**2 - epsilon_g**2)

def find_D_terms(launch_angular_freq: float, K_magnitude: ArrayLike, epsilon_para: ArrayLike, epsilon_perp: ArrayLike, epsilon_g: ArrayLike, theta_m: ArrayLike) -> List[ArrayLike]:
    r"""Returns `[D_11, D_22, D_bb, D_12, D_1b]`"""
    K0 = angular_frequency_to_wavenumber(launch_angular_freq)
    N = K_magnitude / K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)

    D_11 = epsilon_perp - N**2 * sin_theta_m**2
    D_22 = epsilon_perp - N**2
    D_bb = epsilon_para - N**2 * cos_theta_m**2
    D_12 = epsilon_g
    D_1b = N**2 * sin_theta_m * cos_theta_m

    return [D_11, D_22, D_bb, D_12, D_1b]

def find_H_Cardano_eigh(launch_angular_freq: float, K_magnitude: ArrayLike, epsilon_para: ArrayLike, epsilon_perp: ArrayLike, epsilon_g: ArrayLike, theta_m: ArrayLike) -> Tuple[FloatArray, ComplexFloatArray]:
    r"""Returns `(eigvals, eigvecs)`"""
    D_11, D_22, D_bb, D_12, D_1b = find_D_terms(launch_angular_freq, K_magnitude, epsilon_para, epsilon_perp, epsilon_g, theta_m)

    if isinstance(K_magnitude, (float, int)): length = 1
    else: length = K_magnitude.size

    D_tensor = np.zeros([length, 3, 3], dtype=np.complex128)
    D_tensor[:, 0, 0] = D_11
    D_tensor[:, 0, 1] = -1j * D_12
    D_tensor[:, 0, 2] = D_1b
    D_tensor[:, 1, 1] = D_22
    # D_tensor[:, 1, 2] = np.zeros_like(D_11)
    D_tensor[:, 2, 2] = D_bb

    eigvals, eigvecs = np.linalg.eigh(D_tensor)

    return np.squeeze(eigvals), np.squeeze(eigvecs)

def find_mode_index_and_ehat(mode_flag: VALID_LAUNCH_MODE_FLAGS, H_Cardanos: FloatArray, ehats: ComplexFloatArray, tol_H: float = 1e-5, tol_O_mode_polarisation: float = 0.25) -> Tuple[int, float, ComplexFloatArray]:
    r"""`H_Cardanos` and `ehats` must be arranged such that `H_Cardano[i]` corresponds
    to `ehats[:,i]`. If more than one `mode_flag` is provided, then `mode_flag[i]` must
    also correspond to `H_cardano[i]`"""
    soln_idxs = np.where(np.abs(H_Cardanos) <= tol_H)[0]
    soln_H = H_Cardanos[soln_idxs]
    soln_ehats = ehats[:, soln_idxs] # note that soln_H[i] still corresponds to soln_ehats[:,i]

    # We have three cases to check for:
    #   (i)   If there are none or three solutions, raise an error
    #   (ii)  If there is one solution, just return that
    #   (iii) If there is more than 1 solution and the user specifies `mode_flag` = 1 or -1,
    # then we have no way of knowing what exactly the user wants (i.e. we don't know
    # if the O-mode or X-mode solution is wanted), so raise an error
    #         Otherwise, check which mode index yields the desired mode

    if soln_idxs.shape in [(0,), (3,)]:
        raise RuntimeError(f"Unable to check which mode index corresponds to O- and X-mode. Found {soln_idxs.shape} solutions!")
    
    elif soln_idxs.shape == (1,):
        mode_idx = soln_idxs[0]
        H = soln_H[0]
        ehat = soln_ehats
    
    elif soln_idxs.shape == (2,) and mode_flag in [1, -1]:
        raise RuntimeError(f"Specific mode to be used not specified (`mode_flag` = {mode_flag}) and unable to determine which mode indices correspond to O- and X-mode")
    
    else: # soln_idxs.shape == (2,) and mode_flag in ["O", "X"]:
        H0, H1 = H_Cardanos[soln_idxs]
        e0, e1 = soln_ehats.T # because soln_ehats.shape == (3,2)
        abs_Re_e0_b = abs(np.real(e0[-1]))
        abs_Re_e1_b = abs(np.real(e1[-1]))

        if abs_Re_e0_b < 1-tol_O_mode_polarisation and abs_Re_e1_b < 1-tol_O_mode_polarisation:
            log.warning(f"Both computed vectors ({e0} and {e1}) are not within the O-mode polarisation tolerance (tol = {tol_O_mode_polarisation})")
            log.warning(f"Setting the vector with the largets b-component to be the O-mode. This may or may not cause issues")
        
        if mode_flag == "O": idx = 0 if abs_Re_e0_b > abs_Re_e1_b else 1
        else: idx = 1 if abs_Re_e0_b > abs_Re_e1_b else 0
        
        mode_idx = soln_idxs[idx]
        H = soln_H[idx]
        ehat = soln_ehats[idx]
    
    return np.squeeze(mode_idx), np.squeeze(H), np.squeeze(ehat)