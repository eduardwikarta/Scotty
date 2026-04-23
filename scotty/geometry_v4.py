from abc import ABC
import json
import numpy as np
import pathlib
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scotty.derivatives import derivative
from scotty.logger_v4 import logging, timer
from scotty.torbeam import Torbeam
from scotty.typing import ArrayLike, FloatArray
from typing import Any, Callable, Literal, Optional, Tuple, Union, cast

log = logging.getLogger()

##################################################
#
# SPLINE FUNCTIONS
#
##################################################

@timer
def _make_rect_spline(
    R_coord, Z_coord, data_array, interp_order_int: int, interp_smoothing: int
) -> Tuple[Callable[[ArrayLike, Any, ArrayLike], FloatArray], RectBivariateSpline]:
    spline = RectBivariateSpline(
        x = R_coord,
        y = Z_coord,
        z = data_array,
        bbox = [None, None, None, None],
        kx = interp_order_int,
        ky = interp_order_int,
        s = interp_smoothing,
    )
    return lambda q_R, _, q_Z: spline(q_R, q_Z, grid=False), spline

def _make_rect_spline_derivatives(
    spline: RectBivariateSpline,
) -> Tuple[
    Callable[[ArrayLike, Any, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, Any, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, Any, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, Any, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, Any, ArrayLike, float, float], FloatArray],
]:
    dpsi_dR = spline.partial_derivative(1, 0)
    dpsi_dZ = spline.partial_derivative(0, 1)
    d2psi_dR2 = spline.partial_derivative(2, 0)
    d2psi_dZ2 = spline.partial_derivative(0, 2)
    d2psi_dRdZ = spline.partial_derivative(1, 1)

    return (
        lambda q_R, _, q_Z, *args: dpsi_dR(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: dpsi_dZ(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dR2(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dZ2(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dRdZ(q_R, q_Z, grid=False),
    )

@timer
def _make_cuboid_spline(
    X_coord, Y_coord, Z_coord, data_array, interp_order_str: str
) -> Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike], RegularGridInterpolator]:

    spline = RegularGridInterpolator(
        points = (X_coord, Y_coord, Z_coord),
        values = data_array,
        method = interp_order_str,
        bounds_error = False,
    )

    return lambda X,Y,Z: spline((X,Y,Z)), spline



##################################################
#
# FIELD CLASSES, CYLINDRICAL
#
##################################################

class MagneticField_Cylindrical(ABC):
    """Abstract base class for cylindrical magnetic field geometries.
    Child classes must implement these methods"""

    R_coord: FloatArray #: Sample locations for the major radius coordinate
    Z_coord: FloatArray #: Sample locations for the vertical coordinate
    poloidalFlux_grid: FloatArray #: Value of the poloidal magnetic flux, :math:`\psi`, on ``(R_coord, Z_coord)``
    # TODO: Include B_R grid, B_T grid, B_Z grids # TO REMOVE

    def B_R(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_T(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Z(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def polflux(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: raise NotImplementedError

    def d_polflux_dR(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float) -> FloatArray: raise NotImplementedError
    def d_polflux_dZ(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dR2(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dZ2(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dRdZ(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float, delta_Z: float) -> FloatArray: raise NotImplementedError

    def B_X(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Y(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def polflux_in_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError

    def magnitude(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`|B|`, the magnitude of the magnetic field"""
        return np.sqrt( self.B_R(R,_,Z)**2 + self.B_T(R,_,Z)**2 + self.B_Z(R,_,Z)**2 )

    def unitvector(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field"""
        magnitude = self.magnitude(R,_,Z)
        vector = np.array( [self.B_R(R,_,Z), self.B_T(R,_,Z), self.B_Z(R,_,Z)] )
        return (vector / magnitude).T
    
    def unitvector_in_cartesian(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field"""
        magnitude = self.magnitude(R,zeta,Z)
        vector = np.array( [self.B_X(R, zeta, Z), self.B_Y(R, zeta, Z), self.B_Z(R, zeta, Z)] )
        return (vector / magnitude).T



class InterpolatedField_Cylindrical(MagneticField_Cylindrical):
    def __init__(
        self,
        R_coord: FloatArray,
        Z_coord: FloatArray,
        B_R: FloatArray,
        B_T: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: int = 5,
        interp_smoothing: int = 0):

        # Defining class attributes for the interpolated grids
        self.R_coord = R_coord
        self.Z_coord = Z_coord
        self.B_R_grid = B_R
        self.B_T_grid = B_T
        self.B_Z_grid = B_Z
        self.poloidalFlux_grid = psi
        self.interp_order = interp_order

        (self._interp_B_R,
         self._spline_B_R,
         duration_B_R_interpolation) = _make_rect_spline(R_coord, Z_coord, B_R, interp_order, interp_smoothing)
        log.debug(f"Interpolating 2D B_R profile took {duration_B_R_interpolation} s")
        
        (self._interp_B_T,
         self._spline_B_T,
         duration_B_T_interpolation) = _make_rect_spline(R_coord, Z_coord, B_T, interp_order, interp_smoothing)
        log.debug(f"Interpolating 2D B_T profile took {duration_B_T_interpolation} s")

        (self._interp_B_Z,
         self._spline_B_Z,
         duration_B_Z_interpolation) = _make_rect_spline(R_coord, Z_coord, B_Z, interp_order, interp_smoothing)
        log.debug(f"Interpolating 2D B_Z profile took {duration_B_Z_interpolation} s")

        (self._interp_polflux,
         self._spline_polflux,
         duration_psi_interpolation) = _make_rect_spline(R_coord, Z_coord, psi, interp_order, interp_smoothing)
        log.debug(f"Interpolating 2D poloidal flux profile took {duration_psi_interpolation} s")

        self._set_poloidal_flux_derivatives(self._spline_polflux)

    def _set_poloidal_flux_derivatives(self, psi_spline):
        try: (self._dpsi_dR,
              self._dpsi_dZ,
              self._d2psi_dR2,
              self._d2psi_dZ2,
              self._d2psi_dRdZ) = _make_rect_spline_derivatives(psi_spline)
        
        except AttributeError:
            # Older versions of SciPy don't have
            # `RectBivariateSpline.partial_derivative`, so fall back
            # to base class implementation of flux derivatives
            self._dpsi_dR = super().d_polflux_dR
            self._dpsi_dZ = super().d_polflux_dZ
            self._d2psi_dR2 = super().d2_polflux_dR2
            self._d2psi_dZ2 = super().d2_polflux_dZ2
            self._d2psi_dRdZ = super().d2_polflux_dRdZ

    # Defining class attributes for B_R, B_T, B_Z, and polflux
    def B_R(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: return self._interp_B_R(R,_,Z)
    def B_T(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: return self._interp_B_T(R,_,Z)
    def B_Z(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: return self._interp_B_Z(R,_,Z)
    def polflux(self, R: ArrayLike, _: Any, Z: ArrayLike) -> FloatArray: return self._interp_polflux(R,_,Z)

    # Defining the class attributes for the first- and second-order derivatives of polflux
    def d_polflux_dR(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float) -> FloatArray: return self._dpsi_dR(R,_,Z, delta_R)
    def d_polflux_dZ(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_Z: float) -> FloatArray: return self._dpsi_dZ(R,_,Z, delta_Z)
    def d2_polflux_dR2(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float) -> FloatArray: return self._d2psi_dR2(R,_,Z, delta_R)
    def d2_polflux_dZ2(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_Z: float) -> FloatArray: return self._d2psi_dZ2(R,_,Z, delta_Z)
    def d2_polflux_dRdZ(self, R: ArrayLike, _: Any, Z: ArrayLike, delta_R: float, delta_Z: float) -> FloatArray: return self._d2psi_dRdZ(R,_,Z, delta_R, delta_Z)

    # Defining additional class attributes to make abstraction easier
    def B_X(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> FloatArray: return self.B_R(R,None,Z)*np.cos(zeta) - self.B_T(R,None,Z)*np.sin(zeta)
    def B_Y(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> FloatArray: return self.B_R(R,None,Z)*np.sin(zeta) + self.B_T(R,None,Z)*np.cos(zeta)
    def polflux_in_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_polflux(np.sqrt(X**2 + Y**2), None, Z)



##################################################
#
# FIELD CLASSES, CARTESIAN
#
##################################################

class MagneticField_Cartesian(ABC):
    """Abstract base class for cartesian magnetic field geometries.
    Child classes must implement these methods"""

    X_coord: FloatArray
    Y_coord: FloatArray
    Z_coord: FloatArray
    # TODO: Include B_X, B_Y, B_Z, psi grids # TO REMOVE

    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError
    def polflux_in_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: raise NotImplementedError

    def d_polflux_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray: raise NotImplementedError
    def d_polflux_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d_polflux_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError

    def d2_polflux_dX2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dY2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dZ2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dXdY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dXdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float) -> FloatArray: raise NotImplementedError
    def d2_polflux_dYdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float) -> FloatArray: raise NotImplementedError
    
    # Declaring abstract methods (child classes must implement this method)
    # for the magnitude and the unit vector of the magnetic field
    def magnitude(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`|B|`, the magnitude of the magnetic field"""
        return np.sqrt( self.B_X(X,Y,Z)**2 + self.B_Y(X,Y,Z)**2 + self.B_Z(X,Y,Z)**2 )
    
    def unitvector(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field"""
        magnitude = self.magnitude(X,Y,Z)
        vector = np.array( [self.B_X(X,Y,Z), self.B_Y(X,Y,Z), self.B_Z(X,Y,Z)] )
        return (vector / magnitude).T
    
    # Declared purely only for abstraction purposes
    def unitvector_in_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self.unitvector(X,Y,Z)



class InterpolatedField_Cartesian(MagneticField_Cartesian):
    def __init__(
        self,
        X_coord: FloatArray,
        Y_coord: FloatArray,
        Z_coord: FloatArray,
        B_X: FloatArray,
        B_Y: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: int = 5):
        
        # Defining class attributes for the interpolated grids
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.Z_coord = Z_coord
        self.B_X_grid = B_X
        self.B_Y_grid = B_Y
        self.B_Z_grid = B_Z
        self.psi_grid = psi
        self.interp_order = interp_order

        (self._interp_B_X,
         self._spline_B_X,
         duration_B_X_interpolation) = _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_X, interp_order)
        log.debug(f"Interpolating 3D B_X profile took {duration_B_X_interpolation} s")
        
        (self._interp_B_Y,
         self._spline_B_Y,
         duration_B_Y_interpolation) = _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_Y, interp_order)
        log.debug(f"Interpolating 3D B_Y profile took {duration_B_Y_interpolation} s")
        
        (self._interp_B_Z,
         self._spline_B_Z,
         duration_B_Z_interpolation) = _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_Z, interp_order)
        log.debug(f"Interpolating 3D B_Z profile took {duration_B_Z_interpolation} s")
        
        (self._interp_polflux,
         self._spline_polflux,
         duration_polflux_interpolation) = _make_cuboid_spline(X_coord, Y_coord, Z_coord, psi, interp_order)
        log.debug(f"Interpolating 3D poloidal flux profile took {duration_polflux_interpolation} s")

        self.grid_coords = self._spline_B_X.grid()
    
    # Defining the class attributes for B_X, B_Y, B_Z, and polflux
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_X(X,Y,Z)
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Y(X,Y,Z)
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_B_Z(X,Y,Z)
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_polflux(X,Y,Z)
    def polflux_in_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray: return self._interp_polflux(X,Y,Z)
    
    # Defining the class attributes for the first- and second-order derivatives of polflux
    def d_polflux_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray:
        return derivative(self.polflux, ("X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X})
    
    def d_polflux_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y})
    
    def d_polflux_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z})
    
    def d2_polflux_dX2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float) -> FloatArray:
        return derivative(self.polflux, ("X", "X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X})
    
    def d2_polflux_dY2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("Y", "Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y})
    
    def d2_polflux_dZ2(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Z", "Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z})
    
    def d2_polflux_dXdY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float) -> FloatArray:
        return derivative(self.polflux, ("X", "Y"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X, "Y": delta_Y})
    
    def d2_polflux_dXdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("X", "Z"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X, "Z": delta_Z})
    
    def d2_polflux_dYdZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float) -> FloatArray:
        return derivative(self.polflux, ("Y", "Z"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y, "Z": delta_Z})



##################################################
#
# CREATING MAGNETIC FIELD
#
##################################################

# TO REMOVE -- see if can find a more elegant solution instead of interp_order_str/int
def create_magnetic_geometry(
    geometry: Literal["cylindrical", "cartesian"],
    find_B_method: Union[str, MagneticField_Cylindrical, MagneticField_Cartesian],
    interp_order_str: str,
    interp_order_int: int,
    interp_smoothing: int,
    magnetic_data_path: Union[str, pathlib.Path],
    input_filename_suffix: str = "",
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs: Optional[dict]
) -> Union[MagneticField_Cylindrical, MagneticField_Cartesian]:
    
    log.debug(f"Reading and creating field profile")

    # If the user passes an interpolated field, then just use that
    if isinstance(find_B_method, (MagneticField_Cylindrical, MagneticField_Cartesian)):
        log.debug(f"Using existing field profile of type `{type(find_B_method)}` passed from `find_B_method`")
        return find_B_method
    
    # Otherwise, check what it should be and interpolate accordingly
    find_B_method = find_B_method.lower()

    # Standardising inputs
    if isinstance(magnetic_data_path, str): magnetic_data_path = pathlib.Path(magnetic_data_path)

    # NOTE: only cylindrical supported
    if find_B_method == "omfit":
        log.debug(f"Using OMFIT JSON Torbeam file for B and poloidal flux")
        if geometry == "cartesian": raise ValueError(f"`find_B_method` = 'omfit' only works for `geometry` = 'cylindrical'")
        topfile = magnetic_data_path / f"topfile{input_filename_suffix}"

        with open(topfile) as f: data = json.load(f)
        R_coord = np.array(data["R"])
        Z_coord = np.array(data["Z"])

        def unflatten(arr):
            """Convert from column-major (TORBEAM, Fortran) to row-major order (Scotty, Python)"""
            return np.asarray(arr).reshape(len(Z_coord), len(R_coord)).T
        
        return InterpolatedField_Cylindrical(
            R_coord=R_coord,
            Z_coord=Z_coord,
            B_R=unflatten(data["Br"]),
            B_T=unflatten(data["Bt"]),
            B_Z=unflatten(data["Bz"]),
            psi=unflatten(data["pol_flux"]),
            interp_order=interp_order_int,
            interp_smoothing=interp_smoothing,
        )
    
    # NOTE: only cylindrical supported
    elif find_B_method == "torbeam":
        log.debug(f"Using Torbeam input files for B and poloidal flux")
        if geometry == "cartesian": raise ValueError(f"`find_B_method` = 'omfit' only works for `geometry` = 'cylindrical'")
        topfile = magnetic_data_path / f"topfile{input_filename_suffix}"
        torbeam = Torbeam.from_file(topfile)

        return InterpolatedField_Cylindrical(
            R_coord=torbeam.R_grid,
            Z_coord=torbeam.Z_grid,
            B_R=torbeam.B_R,
            B_T=torbeam.B_T,
            B_Z=torbeam.B_Z,
            psi=torbeam.psi,
            interp_order=interp_order_int,
            interp_smoothing=interp_smoothing,
        )
    
    else: raise ValueError(f"Invalid `find_B_method` = '{find_B_method}'")