import numpy as np
from .utils import keV2Angstroms, J4M

def compute_q_map(x, y, x0=0, y0=0, z0=90_000, tx=0, ty=0, keV=10, z_off=0):
    """Compute the momentum-transfer (*q*) map for a detector pixel grid.

    Geometry follows J. Chem. Phys. 113, 9140 (2000).

    Parameters
    ----------
    x, y : np.ndarray
        Pixel coordinates in microns (same shape; e.g. ``J4M.x``, ``J4M.y``).
    x0, y0 : float, optional
        Beam-center coordinates in microns.  Default is 0.
    z0 : float, optional
        Sample-to-detector distance in microns.  Default is 90 000.
    tx, ty : float, optional
        Detector tilt angles in degrees.  Default is 0.
    keV : float, optional
        Photon energy in keV.  Default is 10.
    z_off : float, optional
        Additional z offset in microns.  Default is 0.

    Returns
    -------
    np.ndarray
        Per-pixel momentum transfer values in inverse Angstroms (same shape
        as *x* and *y*).

    Raises
    ------
    ValueError
        If *x* and *y* do not have the same shape, or if *keV* is not positive.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"'x' and 'y' must have the same shape, got {x.shape} and {y.shape}.")
    if keV <= 0:
        raise ValueError(f"'keV' must be positive, got {keV}.")
    tx_rad, ty_rad = np.deg2rad(tx), np.deg2rad(ty)
    z_total = z0 + z_off

    A = -np.sin(ty_rad) * np.cos(tx_rad)
    B = -np.sin(tx_rad)
    C = -np.cos(ty_rad) * np.cos(tx_rad)
    a = x0 + z_total * np.tan(ty_rad)
    b = y0 - z_total * np.tan(tx_rad)
    c = z_total

    R = np.sqrt((x - a) ** 2 + (y - b) ** 2 + c ** 2)
    theta = np.arccos((A * (x - a) + B * (y - b) - C * c) / R)
    lam = keV2Angstroms(keV)
    return 4 * np.pi / lam * np.sin(theta / 2)

def azimuthalBinning(
        img,
        x,
        y,
        x0 = 0,
        y0 = 0,
        z0 = 90000,
        tx = 0,
        ty = 0,
        keV = 10,
        pPlane = 0,
        threshADU = [0,np.inf],
        threshRMS = None,
        mask = None,
        qBin = 0.05,
        rBin = None,
        phiBins = 1,
        geomCorr = True,
        polCorr = True,
        darkImg = None,
        gainImg = None,
        z_off = 0,
        square = False,
        debug = False
    ):
    """Performs azimuthal binning of a 2D image.

    This function integrates a 2D detector image into a 1D profile as a
    function of a radial coordinate (either momentum transfer 'q' or
    real-space radius 'r'). It can perform this integration in multiple
    azimuthal sectors ('phi'). The geometry of the setup, including detector
    distance, tilt, and beam center, is taken into account. Optional
    corrections for polarization are also provided.

    Parameters
    ----------
    img : np.ndarray
        The input image data to be binned.
    x, y : np.ndarray
        Arrays of the same shape as `img` representing the pixel
        x and y coordinates. Default for Jungfrau4M is in micron.
    x0, y0 : float
        x and y-coordinates of the beam center in detector coordinates (J4M: micron). Default is 0.
    z0 : float
        Sample-to-detector distance in detector coordinates (J4M: micron). Default is 90000.
    tx, ty : float, optional
        Detector tilt angles around the y and x axes, respectively, in degrees.
        Default is 0.
    kev : float, optional
        Photon energy of the incident beam in keV. Required for q-space
        binning. Default is 10.
    p_plane : {0, 1}, optional
        The polarization plane of the incident beam.
        1 for vertical polarization, 0 for horizontal. Default is 0.
    threshADU : tuple(float, float), optional
        (min, max) threshold in ADU. Pixel values outside this range are
        set to 0 for the binning calculation. Default is (0, np.inf).
    threshRMS : float, optional
        RMS threshold. Pixel values above this are set to 0. Default is None.
    mask : np.ndarray, optional
        A boolean or integer array of the same shape as `img`. A value of
        True or 1 indicates a pixel to be excluded from the analysis. If none,
        no pixels are excluded. Default is None.
    qBin : float or array_like, optional
        If a float, it's the size of each q-bin in inverse Angstroms (Å⁻¹).
        If an array, it specifies the bin edges for non-uniform binning.
        This parameter is ignored if `r_bin` is specified. Default is 0.05 Å⁻¹.
    rBin : float or array_like, optional
        If specified, binning is performed in real-space radius 'r' (in pixels)
        instead of q-space. If a float, it is the bin size. If an array,
        it specifies the bin edges. Default is None.
    phiBins : int or array_like, optional
        If an int, it's the number of uniform azimuthal bins.
        If an array, it specifies the bin edges in radians for non-uniform
        azimuthal sectors. Default is 1 (no azimuthal binning).
    geom_corr : bool, optional
        If True, apply a geometric (solid angle) correction. Default is True.
    pol_corr : bool, optional
        If True, apply a polarization (Thompson) correction. Default is True.
    dark_img : np.ndarray, optional
        A dark image to be subtracted from `img`. Default is None.
    gain_img : np.ndarray, optional
        A gain/flat-field image to divide `img` by. Default is None.
    z_off : float or np.ndarray, optional
        An additional offset along the beam direction (z-axis) in pixels.
        Default is 0.
    square : bool, optional
        If True, the image is squared before binning. Default is False.
    debug : bool, optional
        If True, print debugging information. Default is False.

    Returns
    -------
    radial_centers : np.ndarray
        1D array of the center values for each radial bin (either q in Å⁻¹
        or r in detector coordinates (J4M: micron).
    azimuthal_average : np.ndarray
        The binned data. A 2D array of shape (`n_phi_bins`, `n_radial_bins`)
        or a 1D array if `phi_bins` is 1.

    Notes
    -----
    - The geometric and angular calculations are based on the methodology
      described in J. Chem. Phys. 113, 9140 (2000).
    - The function preserves the original's specific, non-standard behavior
      of placing pixels that fall outside the defined bin ranges into the
      first bin.
    - The normalization (pixel count per bin) includes all pixels, but the
      intensity summation only includes unmasked pixels. This matches the
      original's logic but may affect the normalization of the first bin if
      a mask is used, as masked pixels are assigned to bin 0 for the count.

    Examples
    --------
    >>> radial_centers, azimuthal_average = azimuthalBinning(img, x, y)
    >>> radial_centers, azimuthal_average = azimuthalBinning(img, x, y, x0=100, y0=150, z0=95000, keV=12.7, qBin=0.02, phiBins=8)
    """
    img, x, y = np.asarray(img, dtype=float), np.asarray(x), np.asarray(y)
    if img.shape != x.shape or img.shape != y.shape:
        raise ValueError(
            f"'img', 'x', and 'y' must have the same shape, got "
            f"{img.shape}, {x.shape}, and {y.shape}."
        )
    if keV <= 0:
        raise ValueError(f"'keV' must be positive, got {keV}.")

    # --- 1. Image Preprocessing ---
    # Apply dark and gain corrections if provided
    if darkImg is not None:
        img = img - darkImg
    if gainImg is not None:
        img = img / gainImg
    if square:
        img = img ** 2
    threshold_mask = (img < threshADU[0]) | (img > threshADU[1])
    if threshRMS is not None:
        threshold_mask |= (img > threshRMS)
    if mask is None:
        mask = np.zeros_like(img, dtype=bool)
    
    # --- 2. Geometric Transformations ---
    tx_rad, ty_rad = np.deg2rad(tx), np.deg2rad(ty)
    z_total = z0 + z_off

    # Geometric parameters from J Chem Phys 113, 9140 (2000)
    A = -np.sin(ty_rad) * np.cos(tx_rad)
    B = -np.sin(tx_rad)
    C = -np.cos(ty_rad) * np.cos(tx_rad)
    a = x0 + z_total * np.tan(ty_rad)
    b = y0 - z_total * np.tan(tx_rad)
    c = z_total

    # Transforming (x,y) to r, theta, phi
    r = np.sqrt((x - a) ** 2 + (y - b) ** 2 + c ** 2)
    matrix_theta = np.arccos((A * (x - a) + B * (y - b) - C * c) / r)
    with np.errstate(invalid='ignore'):
        matrix_phi = np.arccos(
                ((A**2 + C**2) * (y - b) - A * B * (x - a) + B * C * c)
                / np.sqrt((A**2 + C**2) * (r**2 - (A * (x - a) + B * (y - b) - C * c) ** 2))
            )
    
    # Correct NaN values and wrap phi to [0, 2pi]
    matrix_phi[(y >= y0) & (np.isnan(matrix_phi))] = 0
    matrix_phi[(y < y0) & (np.isnan(matrix_phi))] = np.pi
    matrix_phi[x < x0] = 2 * np.pi - matrix_phi[x < x0]

    # --- 3 Correction Factor Calculations ---
    # Default to ones if no correction is applied
    geom_correction = np.ones_like(img, dtype=float)
    pol_correction = np.ones_like(img, dtype=float)

    if geomCorr:
        # Solid angle correction.
        geom_correction = (z_total / r)**3
        # geom_correction /= np.nanmax(geom_correction)

    if polCorr:
        # Polarization or Thompson correction. This is a mixing term, not just a pure polarization correction.
        Pout = 1 - pPlane    
        pol_correction = Pout * (
            1 - (np.sin(matrix_phi) * np.sin(matrix_theta)) ** 2
        ) + pPlane * (1 - (np.cos(matrix_phi) * np.sin(matrix_theta)) ** 2)

    correction = geom_correction * pol_correction

    # --- 4. Binning Setup ---
    # Azimuthal, (phi) binning
    if isinstance(phiBins, (list, np.ndarray)):
        phi_edges = np.sort(np.asarray(phiBins))
        # Ensure range is fully covered for dgitization
        if phi_edges.max() < (2 * np.pi - 0.01):
            phi_edges = np.append(phi_edges, phi_edges.max() + 0.001)
        if phi_edges.min() > 0:
            phi_edges = np.insert(phi_edges, 0, phi_edges.min() - 0.001)
        n_phi_bins = len(phi_edges) - 1
    else:
        n_phi_bins = phiBins
        phi_min, phi_max = np.nanmin(matrix_phi), np.nanmax(matrix_phi)
        phi_edges = np.linspace(phi_min, phi_max, n_phi_bins + 1)

    # Radial (q or r) binning
    if rBin is not None:
        # Binning in real-space radius (r)
        radial_map = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        r_min = np.nanmin(radial_map[~mask])
        r_max = np.nanmax(radial_map[~mask])

        if np.isscalar (rBin):
            if debug: print("r-bin size given: rmax: ", r_max, " rBin ", rBin)
            radial_edges = np.arange(r_min - rBin, r_max + rBin, rBin)
        else:
            radial_edges = np.asarray(rBin)
        radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
        n_radial_bins = len(radial_centers)
    else:
        # Binning in reciprocal space (q)
        radial_map = compute_q_map(x, y, x0, y0, z0, tx, ty, keV, z_off)
        q_min = np.nanmin(radial_map[~mask])
        q_max = np.nanmax(radial_map[~mask])

        if np.isscalar(qBin):
            if debug: print("q-bin size given: qmax: ", q_max, " qBin ", qBin)
            radial_edges = np.arange(0, q_max + qBin, qBin)
        else:
            radial_edges = np.asarray(qBin)
        radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
        n_radial_bins = len(radial_centers)

    # --- 5. Binning Assignment ---
    # Shift phi map slightly to handle edge cases
    phi_step = (phi_edges[1] - phi_edges[0]) / 2 if len(phi_edges) > 1 else 0
    phi_shifted = (matrix_phi + phi_step) % (2 * np.pi)

    phi_indices = np.digitize(phi_shifted.ravel(), phi_edges) - 1
    radial_indices = np.digitize(radial_map.ravel(), radial_edges) - 1

    # Overflow/Underflow handling: put out-of-bounds into first bin
    phi_indices[phi_indices < 0] = 0
    phi_indices[phi_indices >= n_phi_bins] = 0
    radial_indices[mask.ravel()] = 0

    # --- 6. Binning and Normalization ---
    # Create a single 1D index for each pixels (phi, radial) combination
    total_bins = n_phi_bins * n_radial_bins
    combined_indices = np.ravel_multi_index((phi_indices, radial_indices), (n_phi_bins, n_radial_bins))

    # Prep data for intensity summation, excluding masked and thresholded pixels
    final_mask = mask.ravel() | threshold_mask.ravel()
    valid_pixels = ~final_mask

    valid_indices = combined_indices[valid_pixels]
    valid_img = img.ravel()[valid_pixels]
    valid_correction = correction.ravel()[valid_pixels]

    pixel_counts = np.bincount(valid_indices, minlength=total_bins)
    norm_map = np.reshape(pixel_counts, (n_phi_bins, n_radial_bins))

    # Calculate the sum of corrected intensities in each bin
    summed_intensity = np.bincount(
        valid_indices,
        weights=valid_img / valid_correction,
        minlength=total_bins,
    )
    intensity_map = np.reshape(summed_intensity, (n_phi_bins, n_radial_bins))

    # Calculate the final azimuthal average
    with np.errstate(invalid='ignore', divide='ignore'):
        azimuthal_average = intensity_map / norm_map

    return np.squeeze(radial_centers), np.squeeze(azimuthal_average)

def create_J4m_integrator(x0=0, y0=0, z0=90_000, tx=0, ty=0, keV=10, z_off=0):
    """Create a pyFAI AzimuthalIntegrator configured for the Jungfrau 4M.

    The detector is built from the actual J4M pixel-center coordinates,
    so the geometry faithfully reproduces the multi-module layout
    (including inter-module gaps).

    Parameters use the same conventions as :func:`azimuthalBinning`
    (distances in microns, angles in degrees, energy in keV).

    Parameters
    ----------
    x0, y0 : float, optional
        Beam-center coordinates in microns.  Default is 0.
    z0 : float, optional
        Sample-to-detector distance in microns.  Default is 90 000.
    tx, ty : float, optional
        Detector tilt angles in degrees.  Default is 0.
    keV : float, optional
        Photon energy in keV.  Default is 10.
    z_off : float, optional
        Additional z offset in microns.  Default is 0.

    Returns
    -------
    pyFAI.integrator.azimuthal.AzimuthalIntegrator
        Configured integrator ready to call ``integrate1d`` / ``integrate2d``.

    Raises
    ------
    ImportError
        If pyFAI is not installed.
    """

    # Lazy load these 
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    dist = (z0 + z_off) * 1e-6            # microns → metres
    poni1 = y0 * 1e-6                      # slow axis = y
    poni2 = x0 * 1e-6                      # fast axis = x
    wavelength = 12.39841984e-10 / keV     # keV → metres
    # pyFAI rot1 = rotation about axis-1 (y), rot2 = rotation about axis-2 (x)
    rot1 = np.deg2rad(ty)
    rot2 = np.deg2rad(tx)

    # --- Create a custom pyFAI Detector with pixel corners from J4M geometry ---

    # Flatten (8, 512, 1024) → (4096, 1024) and convert µm → m
    xc = J4M.x.reshape(-1, J4M.x.shape[-1]) * 1e-6
    yc = J4M.y.reshape(-1, J4M.y.shape[-1]) * 1e-6
    n_slow, n_fast = xc.shape

    half = 37.5e-6  # half-pixel in meters

    # pyFAI corner array: (n_slow, n_fast, 4, 3) — coords are (z, y, x)
    corners = np.zeros((n_slow, n_fast, 4, 3), dtype=np.float64)
    for idx, (dy, dx) in enumerate([
        (-half, -half),   # corner 0: bottom-left
        (-half, +half),   # corner 1: bottom-right
        (+half, +half),   # corner 2: top-right
        (+half, -half),   # corner 3: top-left
    ]):
        corners[:, :, idx, 1] = yc + dy   # y
        corners[:, :, idx, 2] = xc + dx   # x
        # z stays 0 — all pixels in the detector plane

    detector = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(n_slow, n_fast))
    detector.set_pixel_corners(corners)

    # --- Create the AzimuthalIntegrator with the custom detector ---

    ai = AzimuthalIntegrator(
        dist=dist,
        poni1=poni1,
        poni2=poni2,
        rot1=rot1,
        rot2=rot2,
        rot3=0,
        wavelength=wavelength,
        detector=detector,
    )
    return ai


def azimuthalBinning_pyfai(
        img,
        x0=0,
        y0=0,
        z0=90_000,
        tx=0,
        ty=0,
        keV=10,
        pPlane=0,
        mask=None,
        qBin=0.05,
        rBin=None,
        phiBins=1,
        polCorr=True,
        geomCorr=True,
        darkImg=None,
        gainImg=None,
        z_off=0,
        method=("full", "CSR", "cython"),
        ai=None,
    ):
    """Azimuthal integration using pyFAI — drop-in alternative to :func:`azimuthalBinning`.

    Accepts the same geometry conventions (microns / degrees / keV) so
    existing parameter dictionaries can be reused.  The image is
    automatically reshaped from the Jungfrau 4M panel layout
    ``(8, 512, 1024)`` to ``(4096, 1024)``.

    Parameters
    ----------
    img : np.ndarray
        Detector image, either ``(8, 512, 1024)`` or ``(4096, 1024)``.
    x0, y0 : float, optional
        Beam-center in microns.  Default is 0.
    z0 : float, optional
        Sample-to-detector distance in microns.  Default is 90,000.
    tx, ty : float, optional
        Detector tilt angles in degrees.  Default is 0.
    keV : float, optional
        Photon energy in keV.  Default is 10.
    pPlane : {0, 1}, optional
        Polarization plane (0 = horizontal, 1 = vertical).  Default is 0.
    mask : np.ndarray or None, optional
        Boolean mask (True = excluded pixel).  Same shape as *img*.
    qBin : float, optional
        Bin width in Å⁻¹ (used to compute the number of radial points).
        Default is 0.05.
    rBin : float or None, optional
        If given, integrate in real-space radius (microns) instead of
        *q*-space.  Default is None.
    phiBins : int, optional
        Number of azimuthal sectors.  ``1`` gives a standard 1-D
        integration; ``> 1`` produces a 2-D cake.  Default is 1.
    polCorr : bool, optional
        Apply polarization correction.  Default is True.
    geomCorr : bool, optional
        Apply solid-angle correction.  Default is True.
    darkImg : np.ndarray or None, optional
        Dark image subtracted before integration.
    gainImg : np.ndarray or None, optional
        Gain / flat-field image (pyFAI divides by this).
    z_off : float, optional
        Additional z offset in microns.  Default is 0.
    method : tuple, optional
        pyFAI integration method descriptor.
        Default ``("full", "CSR", "cython")`` uses full pixel splitting
        with a CSR sparse-matrix engine — fast after the first call.
    ai : AzimuthalIntegrator or None, optional
        A pre-built integrator from :func:`create_pyfai_integrator`.
        When integrating many images with the same geometry, pass the
        same *ai* object to avoid rebuilding the lookup table each time.
        If *None*, a new integrator is created from the geometry
        parameters.

    Returns
    -------
    radial : np.ndarray
        Bin centers in Å⁻¹ (or microns if *rBin* is set).
    intensity : np.ndarray
        Integrated intensity.  1-D when *phiBins* is 1, otherwise
        2-D of shape ``(phiBins, n_radial)``.
    """
    if ai is None:
        ai = create_J4m_integrator(x0, y0, z0, tx, ty, keV, z_off)

    # --- reshape multi-panel images ---
    img = np.asarray(img, dtype=float)
    if img.ndim == 3 and img.shape[0] == 8:
        img = img.reshape(-1, img.shape[-1])
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[0] == 8:
            mask = mask.reshape(-1, mask.shape[-1])
    if darkImg is not None:
        darkImg = np.asarray(darkImg, dtype=float)
        if darkImg.ndim == 3 and darkImg.shape[0] == 8:
            darkImg = darkImg.reshape(-1, darkImg.shape[-1])
    if gainImg is not None:
        gainImg = np.asarray(gainImg, dtype=float)
        if gainImg.ndim == 3 and gainImg.shape[0] == 8:
            gainImg = gainImg.reshape(-1, gainImg.shape[-1])

    # --- polarization factor ---
    # pyFAI convention: 1.0 = fully horizontal, -1.0 = fully vertical, 0 = unpolarized
    pol_factor = (1 - 2 * pPlane) if polCorr else None

    # --- determine unit and bin count ---
    if rBin is not None:
        unit = "r_mm"
        # r_mm uses millimetres; our rBin is in microns
        r_max_mm = np.sqrt(img.shape[0] * 75e-3 ** 2 + img.shape[1] * 75e-3 ** 2) 
        npt = max(int(r_max_mm / (rBin * 1e-3)), 10)
    else:
        unit = "q_A^-1"
        lam = keV2Angstroms(keV)
        # generous upper-bound for q
        q_max = 4 * np.pi / lam
        npt = max(int(q_max / qBin), 10)

    # --- integrate ---
    if phiBins == 1:
        result = ai.integrate1d(
            img,
            npt,
            unit=unit,
            mask=mask,
            dark=darkImg,
            flat=gainImg,
            polarization_factor=pol_factor,
            correctSolidAngle=geomCorr,
            method=method,
        )
        return result.radial, result.intensity
    else:
        result = ai.integrate2d(
            img,
            npt,
            phiBins,
            unit=unit,
            mask=mask,
            dark=darkImg,
            flat=gainImg,
            polarization_factor=pol_factor,
            correctSolidAngle=geomCorr,
            method=method,
        )
        return result.radial, result.intensity