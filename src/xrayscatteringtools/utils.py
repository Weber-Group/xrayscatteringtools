import re
from IPython import get_ipython
import numpy as np

def enable_underscore_cleanup():
    """
    Register a Jupyter Notebook post-cell hook to automatically delete user-defined 
    single-underscore variables after each cell execution.

    Notes
    -----
    This helps keep the notebook namespace clean by removing temporary variables
    that start with a single underscore (`_`) but are not standard IPython variables 
    (like `_i`, `_oh`, etc.) or Python "dunder" variables.
    """
    ipython = get_ipython()
    user_ns = ipython.user_ns  # This gives you access to the Jupyter notebook namespace

    def clean_user_underscore_vars(*args, **kwargs):
        def is_user_defined_underscore(var):
            return (
                var.startswith('_')
                and not re.match(r'^_i\d*$|^_\d*$|^_ih$|^_oh$|^_ii*$|^_iii$|^_dh$|^_$', var)
                and not var.startswith('__')
            )

        for var in list(user_ns):
            if is_user_defined_underscore(var):
                del user_ns[var]

    ipython.events.register('post_run_cell', clean_user_underscore_vars)

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
        geom_correction = (z_total / r**3)
        geom_correction /= np.nanmax(geom_correction)

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
        lam = keV2Angstroms(keV)
        radial_map = 4 * np.pi / lam * np.sin(matrix_theta / 2)
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
    # Create a signle 1D index for each pixels (phi, radial) combination
    total_bins = n_phi_bins * n_radial_bins
    combined_indices = np.ravel_multi_index((phi_indices, radial_indices), (n_phi_bins, n_radial_bins))

    indicies_for_norm = combined_indices.copy()
    indicies_for_norm[mask.ravel()] = 0  # Masked pixels go to first bin
    pixel_counts = np.bincount(indicies_for_norm, minlength=total_bins)
    norm_map = np.reshape(pixel_counts, (n_phi_bins, n_radial_bins))

    # Prep data for intensity summation, excluding masked and thresholded pixels
    final_mask = mask.ravel() | threshold_mask.ravel()
    valid_pixels = ~final_mask

    valid_indices = combined_indices[valid_pixels]
    valid_img = img.ravel()[valid_pixels]
    valid_correction = correction.ravel()[valid_pixels]

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

def keV2Angstroms(keV):
    """
    Convert photon energy from keV to wavelength in Angstroms.

    Parameters
    ----------
    keV : float
        Photon energy in kilo-electron volts.

    Returns
    -------
    float
        Corresponding wavelength in Angstroms.

    Notes
    -----
    Uses the relation λ(Å) = 12.39841984 / E(keV).
    """
    return 12.39841984/keV

def Angstroms2keV(angstroms):
    """
    Convert wavelength in Angstroms to photon energy in keV.

    Parameters
    ----------
    angstroms : float
        Wavelength in Angstroms.

    Returns
    -------
    float
        Photon energy in kilo-electron volts.

    Notes
    -----
    Uses the relation E(keV) = 12.39841984 / λ(Å).
    """
    return 12.39841984/angstroms

def q2theta(q, keV):
    """
    Convert momentum transfer q to scattering angle theta in radians.

    Parameters
    ----------
    q : float or array-like
        Momentum transfer in inverse Angstroms (Å⁻¹).
    keV : float
        Photon energy in keV.

    Returns
    -------
    float or array-like
        Scattering angle θ in radians.

    Notes
    -----
    Uses the relation θ = 2 * arcsin(q * λ / (4π)), where λ is the photon
    wavelength corresponding to the given energy.
    """
    return 2 * np.arcsin(q * keV2Angstroms(keV) / (4 * np.pi))

def theta2q(theta, keV):
    """
    Convert scattering angle theta in radians to momentum transfer q.

    Parameters
    ----------
    theta : float or array-like
        Scattering angle in radians.
    keV : float
        Photon energy in keV.

    Returns
    -------
    float or array-like
        Momentum transfer q in inverse Angstroms (Å⁻¹).

    Notes
    -----
    Uses the relation q = (4π / λ) * sin(θ / 2), where λ is the photon wavelength
    corresponding to the given energy.
    """
    return 4 * np.pi / keV2Angstroms(keV) * np.sin(theta / 2)

ELEMENT_NUMBERS = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,
    "C": 6,  "N": 7,   "O": 8,   "F": 9,   "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53,  "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,  "Np": 93, "Pu": 94, "Am": 95,
    "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115,
    "Lv": 116, "Ts": 117, "Og": 118
}

ELEMENT_SYMBOLS = {v: k for k, v in ELEMENT_NUMBERS.items()}


def element_symbol_to_number(symbol: str) -> int:
    """
    Convert an element symbol (e.g. 'O') to its atomic number (e.g. 8).
    
    Parameters
    ----------
    symbol : str
        The element symbol, case-sensitive (e.g. 'H', 'He', 'Fe').

    Returns
    -------
    int
        Atomic number of the element.

    Raises
    ------
    KeyError
        If the symbol is not valid.
    """
    return ELEMENT_NUMBERS[symbol]


def element_number_to_symbol(number: int) -> str:
    """
    Convert an atomic number (e.g. 8) to its element symbol (e.g. 'O').
    
    Parameters
    ----------
    number : int
        The atomic number (1 to 118).

    Returns
    -------
    str
        Element symbol.

    Raises
    ------
    KeyError
        If the atomic number is not valid.
    """
    return ELEMENT_SYMBOLS[number]