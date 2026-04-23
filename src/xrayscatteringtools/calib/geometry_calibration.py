import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from ..utils import theta2q
from .scattering_corrections import correction_factor, forward_scattering_correction

def run_geometry_calibration(
        raw_image,
        x,
        y,
        mask,
        theory_q,
        theory_Iq,
        photon_energy_keV,
        initial_guess={'amplitude' : 1,'x0': 0, 'y0': 0, 'z0': 90000},
        polarization = 0,
        mask_center=True,
        mask_center_size=7500,
        bounds = ([0, -10_000, -10_000, 30_000], [np.inf, 10_000, 10_000, 150_000])
        ):
    """
    Perform geometry calibration on a raw detector image using a theoretical scattering pattern.

    This function fits the detector geometry parameters (x0, y0, z0) by comparing the
    measured image to a theoretical scattering pattern and applying corrections for
    polarization (Thompson), geometry, and angle-of-scattering effects.

    Parameters
    ----------
    raw_image : ndarray
        Array of measured intensities from the detector.
    x : ndarray
        Array of x-coordinates for each pixel.
    y : ndarray
        Array of y-coordinates for each pixel.
    mask : ndarray of bool
        Boolean mask array indicating which pixels to include in the fit.
    theory_q : ndarray
        1D array of scattering vector magnitudes corresponding to the theoretical pattern.
    theory_Iq : ndarray
        1D array of theoretical scattering intensities at each q.
    photon_energy_keV : float
        Photon energy of the X-ray in keV.
    initial_guess : dict, optional
        Initial guess for fit parameters:
        - 'amplitude' : float, scaling of the intensity
        - 'x0', 'y0', 'z0' : float, initial guesses for detector geometry
        Default: {'amplitude': 1, 'x0': 0, 'y0': 0, 'z0': 90000}.
    polarization : float, optional
        Polarization direction (-pi/2 to pi/2). Default is 0 (horizontally polarized).
    mask_center : bool, optional
        If True, exclude pixels near the center from the fit. Default is True.
    mask_center_size : float, optional
        Radius (in pixels) of the excluded central region. Default is 7500.
    fit_bounds : tuple of array_like, optional
        Lower and upper bounds for the fit parameters. Default is 
        ([0, -10000, -10000, 30000, -π/2], [inf, 10000, 10000, 150000, π/2]).

    Returns
    -------
    fit : ndarray
        3D array representing the fitted detector image (reshaped to match input dimensions).
    popt : ndarray
        Optimized fit parameters: [amplitude, x0, y0, z0]. In units of microns.
    pcov : ndarray
        Covariance matrix of the optimized parameters.

    Notes
    -----
    The returned z0 distance must be converted to millimeters when editing the producer. It should not be in microns.
    The fitting applies:
    - `thompson_correction` for polarization effects,
    - `geometry_correction` for geometric distortions,
    - `correction_factor` for angle-of-scattering corrections (Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602).

    Examples
    --------
    >>> fit, popt, pcov = run_geometry_calibration(raw_image, x, y, mask, theory_q, theory_Iq, 12.7)
    """
    # Step one: generate an interpolation for the theory pattern.
    theory_interpolation = InterpolatedUnivariateSpline(theory_q, theory_Iq, ext=3) # Extrapolation 3. Returns Boundary value if outside of range.

    # Step two: Define initial guess, bounds, and fitting function using a wrapper
    p0 = [initial_guess['amplitude'], initial_guess['x0'], initial_guess['y0'], initial_guess['z0']]
    def fitting_function(xy, amplitude, x0, y0, z0):
        return model(xy, amplitude, x0, y0, z0, polarization, photon_energy_keV, theory_interpolation)

    # Step three: Masking and formatting variables
    center_mask = np.ones_like(raw_image, dtype=bool)
    if mask_center:
        center_mask[np.sqrt(x**2 + y**2) < mask_center_size] = False
        
    masked_data = np.ravel(raw_image[mask & center_mask])
    x_masked = np.ravel(x[mask & center_mask])
    y_masked = np.ravel(y[mask & center_mask])
    xy_masked = [x_masked, y_masked]

    # Step four: fitting the data
    popt, pcov = curve_fit(fitting_function, xy_masked, masked_data, p0=p0, bounds=bounds)
    fit = fitting_function([np.ravel(x), np.ravel(y)], *popt).reshape(raw_image.shape)
    return fit, popt, pcov

def model(
        xy,
        amplitude,
        x0,
        y0,
        z0,
        phi0,
        photon_energy_keV,
        theory_interpolation,
        do_geometry_correction=True,
        do_thompson_correction=True,
        do_angle_of_scattering_correction=True,
        do_geometry_correction_units=False,
        dx=75,
        dy=75
        ):
    """
    Calculate the theoretical detector image for given geometry parameters.

    Parameters
    ----------
    xy : list of ndarray
        [x, y] 2D arrays of pixel coordinates.
    amplitude : float
        Scaling factor for intensity.
    x0, y0, z0 : float
        Detector geometry parameters (pixel offsets and distance).
    phi0 : float
        Azimuthal rotation angle in radians.
    photon_energy_keV : float
        Photon energy in keV.
    theory_interpolation : callable
        Interpolated function for theoretical scattering intensities versus q.
    do_geometry_correction : bool, optional
        If True, apply geometry correction. Default is True.
    do_thompson_correction : bool, optional
        If True, apply Thompson polarization correction. Default is True.
    do_angle_of_scattering_correction : bool, optional
        If True, apply angle-of-scattering correction. Default is True.
    do_geometry_correction_units : bool, optional
        If True, apply geometry accounting for proper solid angle subtension. Default is False.
    dx, dy : float, optional
        Pixel size in microns for geometry correction with units. Default is 75 microns. If do_geometry_correction_units is False, these are ignored.

    Returns
    -------
    fit : ndarray
        Flattened array of predicted intensities for each pixel.

    Notes
    -----
    Do not use both do_geometry_correction and do_geometry_correction_units at the same time.
    """
    # Pull out the x and y arrays from the input
    x = xy[0]
    y = xy[1]
    # Center the arrays around x0 and y0
    centered_x = x - x0
    centered_y = y - y0
    # Calculate the r array
    r_matrix = np.sqrt(centered_x**2 + centered_y**2)
    theta_matrix = np.arctan(r_matrix/z0)
    q_matrix = theta2q(theta_matrix, photon_energy_keV)

    # Calculate corrections
    corrections = np.ones_like(q_matrix)
    if do_thompson_correction:
        corrections *= thompson_correction(centered_x, centered_y, z0, phi0) # Polarization
    if do_geometry_correction:
        corrections *= geometry_correction(centered_x, centered_y, z0) # Geometry
    if do_angle_of_scattering_correction:
        corrections *= correction_factor(q_matrix, photon_energy_keV) # Angle-Of-Scattering, Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602
    if do_geometry_correction_units:
        corrections *= geometry_correction_units(centered_x, centered_y, z0, dx, dy) # Geometry correctly accounting for solid angle subtension per pixel
    fit = amplitude * theory_interpolation(q_matrix) * corrections
    return fit

def thompson_correction(x, y, z0, phi0):
    """
    Calculate the Thompson polarization correction for each pixel.

    Parameters
    ----------
    x, y : ndarray
        Pixel coordinates relative to signal origin.
    z0 : float
        Detector distance along beam axis.
    phi0 : float
        Azimuthal rotation angle in radians.

    Returns
    -------
    correction : ndarray
        Thompson correction factor for each pixel.

    Notes
    -----
    See Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602

    """
    # Calculate the Thompson scattering correction factor
    r_matrix = np.sqrt(x**2 + y**2)
    theta = np.arctan(r_matrix/z0)
    phi = np.arctan2(y, x) + phi0
    correction = np.sin(phi)**2+np.cos(theta)**2*np.cos(phi)**2
    return correction

def geometry_correction(x, y, z0):
    """
    Compute geometric correction factor (cos^3(theta)) for a detector.

    Parameters
    ----------
    x, y : ndarray
        Pixel coordinates relative to signal origin.
    z0 : float
        Detector distance along beam axis.

    Returns
    -------
    correction : ndarray
        Geometric correction factor for each pixel.

    Notes
    -----
    The geometry correction comes from the inverse square law and the effective area of the pixel.
    The inverse square law accounts for cos^2(theta), and the effective area is another cos(theta).
    See Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602


    """
    r_matrix = np.sqrt(x**2 + y**2)
    theta = np.arctan(r_matrix / z0)
    correction = np.cos(theta) ** 3
    return correction

def geometry_correction_units(x, y, z0, dx, dy):
    """
    Compute geometric correction factor z0^2 cos^3(theta) / dxdy for a detector, accounting for pixel area and distance units.
    
    Parameters
    ----------
    x, y : ndarray
        Pixel coordinates relative to signal origin.
    z0 : float
        Detector distance along beam axis.
    dx, dy : float
        Pixel size in x and y directions.
    
    Returns
    -------
    correction : ndarray
        Geometric correction factor for each pixel.

    Notes
    -----
    The geometry correction comes from the inverse square law and the effective area of the pixel.
    The inverse square law accounts for z^2 cos^2(theta), and the effective area is cos(theta)/dxdy.
    """
    r_matrix = np.sqrt(x**2 + y**2)
    theta = np.arctan(r_matrix / z0)
    correction = np.cos(theta)**3 * (dx * dy) / z0**2
    return correction


def run_geometry_calibration_full(
        raw_image,
        x,
        y,
        mask,
        theory_q,
        theory_I_total,
        photon_energy_keV,
        formula,
        initial_guess={'amplitude': 1, 'x0': 0, 'y0': 0, 'z0': 90000},
        polarization=0,
        mask_center=True,
        mask_center_size=7500,
        bounds=([0, -10_000, -10_000, 30_000], [np.inf, 10_000, 10_000, 150_000]),
        dx=75,
        dy=75,
        L=2.4e-3,
        t_Pt=125e-6,
        t_Pt_countersink=75e-6,
        r_Pt_entry=300e-6,
        r_Pt_bore=125e-6,
        t_Be=100e-6,
        r_Be=125e-6,
        t_K=8e-6,
        t_Al=4.5e-6,
        t_Si=318.5e-6,
        theta_max=None,
        n_theta=256,
        n_z=128,
        n_EF=80,
        EF_low_keV=None,
        ):
    """
    Geometry calibration using a full forward scattering / detection model.

    Unlike `run_geometry_calibration`, this routine:

    - takes the *total* theory pattern I_total(q) (elastic + Compton) at any
      level of theory, plus the molecular formula, and derives the Compton
      fraction internally using IAM atomic tables;
    - integrates over the scattering position z_s along the Pt bore, gas cell,
      and Be bore (uniform density assumed), so the Be-hole vs Be-material
      geometry is handled per-z_s rather than with a single effective cut;
    - models the Pt pinhole as fully opaque with a countersunk geometry (the
      downstream straight-bore rim binds at forward angles);
    - resolves the energy dependence of Be, Kapton, Al, and Si attenuation
      separately for elastic (at EI) and Compton (integrated over the
      double-differential profile J(theta, E_F) from `iam_compton_spectrum`);
    - applies deposited-energy weighting (E_F/EI) to the Compton channel for
      the Jungfrau4M operated in normal-gain integrating mode.

    The per-pixel forward model is
        I(x, y) = amplitude * thompson(theta, phi)
                            * (dx * dy * cos^3(theta) / z0^2)
                            * I_total(q(theta)) * C(theta)
    where C(theta) is the tabulated correction returned by
    `forward_scattering_correction` and is normalized so C(0) = 1.

    Parameters
    ----------
    raw_image, x, y, mask : ndarray
        Measured image, pixel x/y (microns), and boolean inclusion mask.
    theory_q : ndarray
        1D q grid (inverse Angstroms) for the theory pattern.
    theory_I_total : ndarray
        Total theory intensity (elastic + Compton) on `theory_q`.
    photon_energy_keV : float
        Incident photon energy in keV.
    formula : str
        Chemical formula of the gas sample (e.g. "SF6"). Used to split the
        total into elastic and Compton fractions and to compute the Compton
        energy distribution J(theta, E_F).
    initial_guess, polarization, mask_center, mask_center_size, bounds :
        Same meaning as in `run_geometry_calibration`.
    dx, dy : float, optional
        Pixel size in microns. Default 75.
    L, t_Pt, t_Pt_countersink, r_Pt_entry, r_Pt_bore, t_Be, r_Be, t_K, t_Al, t_Si :
        Cell / window / detector geometry (SI meters). Defaults match the
        user's gas-cell setup.
    theta_max : float, optional
        Upper bound of the theta tabulation. If None, inferred from the
        pixel positions and the lower z0 bound.
    n_theta, n_z, n_EF, EF_low_keV :
        Numerics passed through to `forward_scattering_correction`.

    Returns
    -------
    fit : ndarray
        Best-fit detector image reshaped to `raw_image.shape`.
    popt : ndarray
        Optimized parameters [amplitude, x0, y0, z0]. z0 is in microns.
    pcov : ndarray
        Covariance matrix of the optimized parameters.
    """
    if theta_max is None:
        max_r = float(np.sqrt(np.max(x ** 2 + y ** 2)))
        z0_lower = bounds[0][3]
        theta_max = min(1.2, float(np.arctan(max_r / z0_lower)) * 1.15)

    C_spline = forward_scattering_correction(
        photon_energy_keV,
        formula,
        theory_q,
        theory_I_total,
        theta_max=theta_max,
        n_theta=n_theta,
        L=L,
        t_Pt=t_Pt,
        t_Pt_countersink=t_Pt_countersink,
        r_Pt_entry=r_Pt_entry,
        r_Pt_bore=r_Pt_bore,
        t_Be=t_Be,
        r_Be=r_Be,
        t_K=t_K,
        t_Al=t_Al,
        t_Si=t_Si,
        n_z=n_z,
        n_EF=n_EF,
        EF_low_keV=EF_low_keV,
    )

    I_total_spline = InterpolatedUnivariateSpline(
        np.asarray(theory_q, dtype=float),
        np.asarray(theory_I_total, dtype=float),
        ext=3,
    )

    p0 = [initial_guess['amplitude'], initial_guess['x0'], initial_guess['y0'], initial_guess['z0']]

    def fitting_function(xy, amplitude, x0, y0, z0):
        return model_full(
            xy, amplitude, x0, y0, z0,
            polarization, photon_energy_keV,
            I_total_spline, C_spline,
            dx=dx, dy=dy,
        )

    center_mask = np.ones_like(raw_image, dtype=bool)
    if mask_center:
        center_mask[np.sqrt(x ** 2 + y ** 2) < mask_center_size] = False

    combined = mask & center_mask
    masked_data = np.ravel(raw_image[combined])
    xy_masked = [np.ravel(x[combined]), np.ravel(y[combined])]

    popt, pcov = curve_fit(fitting_function, xy_masked, masked_data, p0=p0, bounds=bounds)
    fit = fitting_function([np.ravel(x), np.ravel(y)], *popt).reshape(raw_image.shape)
    return fit, popt, pcov


def model_full(
        xy,
        amplitude,
        x0,
        y0,
        z0,
        phi0,
        photon_energy_keV,
        I_total_spline,
        C_spline,
        dx=75.0,
        dy=75.0,
        ):
    """
    Per-pixel forward model used by `run_geometry_calibration_full`.

    Combines the precomputed total theory pattern with the precomputed
    scattering / detection correction C(theta), plus the Thompson polarization
    factor and the pixel solid-angle factor cos^3(theta) * dx*dy / z0^2.

    Parameters
    ----------
    xy : list of ndarray
        [x, y] pixel coordinates (same units as x0, y0, z0, dx, dy).
    amplitude, x0, y0, z0 : float
        Scaling and detector geometry parameters.
    phi0 : float
        Polarization azimuth in radians.
    photon_energy_keV : float
        Incident photon energy in keV.
    I_total_spline : callable
        Total theory intensity as a function of q (inverse Angstroms).
    C_spline : callable
        Angle-dependent correction C(theta) from
        `forward_scattering_correction`.
    dx, dy : float, optional
        Pixel size. Default 75 microns.

    Returns
    -------
    fit : ndarray
        Flattened predicted intensities for each pixel.
    """
    x = xy[0]
    y = xy[1]
    cx = x - x0
    cy = y - y0
    r_matrix = np.sqrt(cx * cx + cy * cy)
    theta = np.arctan(r_matrix / z0)
    q = theta2q(theta, photon_energy_keV)

    thomson = thompson_correction(cx, cy, z0, phi0)
    solid_angle = np.cos(theta) ** 3 * (dx * dy) / (z0 * z0)

    return amplitude * thomson * solid_angle * I_total_spline(q) * C_spline(theta)