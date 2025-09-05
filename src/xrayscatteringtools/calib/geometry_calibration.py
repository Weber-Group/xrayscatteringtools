import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from xrayscatteringtools.utils import theta2q
from xrayscatteringtools.calib.scattering_corrections import correction_factor

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

def model(xy, amplitude, x0, y0, z0, phi0, photon_energy_keV, theory_interpolation):
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

    Returns
    -------
    fit : ndarray
        Flattened array of predicted intensities for each pixel.

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
    corrections *= thompson_correction(centered_x, centered_y, z0, phi0) # Polarization
    corrections *= geometry_correction(centered_x, centered_y, z0) # Geometry
    corrections *= correction_factor(q_matrix, photon_energy_keV) # Angle-Of-Scattering, Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602
    fit = amplitude * theory_interpolation(q_matrix) * corrections
    return fit

def thompson_correction(x, y, z0, phi0):
    """
    Calculate the Thompson polarization correction for each pixel.

    Parameters
    ----------
    x, y : ndarray
        Pixel coordinates relative to detector origin.
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
        Pixel coordinates relative to detector origin.
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