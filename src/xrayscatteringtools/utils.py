import re
from IPython import get_ipython
import numpy as np
import h5py
from types import SimpleNamespace
import pathlib

_data_path = pathlib.Path(__file__).parent / "data"

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

def au2invAngstroms(au):
    """
    Convert momentum transfer from atomic units (a.u., 1/Bohr) to inverse Angstroms (Å⁻¹).

    Parameters
    ----------
    au : float
        Momentum transfer in atomic units (1/Bohr).

    Returns
    -------
    float
        Corresponding momentum transfer in inverse Angstroms.

    Notes
    -----
    Uses the conversion factor 1 a.u. = 1.8897261259077822 Å⁻¹.
    """
    return 1.8897261259077822 * au

def invAngstroms2au(invA):
    """
    Convert momentum transfer from inverse Angstroms (Å⁻¹) to atomic units (a.u., 1/Bohr).

    Parameters
    ----------
    invA : float
        Momentum transfer in inverse Angstroms (Å⁻¹).

    Returns
    -------
    float
        Corresponding momentum transfer in atomic units (1/Bohr).

    Notes
    -----
    Uses the conversion factor 1 a.u. = 1.8897261259077822 Å⁻¹.
    """
    return invA / 1.8897261259077822

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

def translate_molecule(coords, translation_vector):
    """
    Translate molecular coordinates by a given vector.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of atomic coordinates.
    translation_vector : np.ndarray
        1x3 array representing the translation vector.

    Returns
    -------
    np.ndarray
        Translated coordinates.
    """
    return coords + translation_vector

def rotate_molecule(coords, alpha, beta, gamma):
    """
    Rotate molecular coordinates by given Euler angles (in degrees).
    Uses the ZYX convention (yaw-pitch-roll).
    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of atomic coordinates.
    alpha : float
        Rotation angle around x-axis in degrees.
    beta : float
        Rotation angle around y-axis in degrees.
    gamma : float
        Rotation angle around z-axis in degrees.

    Returns
    -------
    np.ndarray
        Rotated coordinates.
    """
    # Convert angles from degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    # Rotation matrices around x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
    
    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    
    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    
    # Rotate all coordinates
    rotated_coords = coords @ R.T
    return rotated_coords

def _load_J4M():
    file_path = _data_path / "Jungfrau4M.h5"
    with h5py.File(file_path, "r") as f:
        # Load all datasets into memory
        data = {k: f[k][()] for k in f.keys()}
        obj = SimpleNamespace(**data)
        obj.__doc__ = """
        Jungfrau4M constant properties.

        Attributes
        ----------
        x : ndarray of shape (8, 512, 1024)
            Pixel x-coordinates in microns.
        y : ndarray of shape (8, 512, 1024)
            Pixel y-coordinates in microns.
        line_mask : Boolean ndarray of shape (8, 512, 1024)
            Line mask for the detector.
        t_mask : Boolean ndarray of shape (8, 512, 1024)
            T-mask for the detector.
        """
    return obj


class _LazyJ4M:
    """Descriptor that defers loading Jungfrau4M.h5 until first attribute access."""
    _instance = None

    def __getattr__(self, name):
        if _LazyJ4M._instance is None:
            _LazyJ4M._instance = _load_J4M()
        return getattr(_LazyJ4M._instance, name)

    def __repr__(self):
        if _LazyJ4M._instance is None:
            return "<J4M: not yet loaded>"
        return repr(_LazyJ4M._instance)


J4M = _LazyJ4M()

def compress_ranges(nums):
    """
    Compress a sequence of integers into a compact range string.

    Given an iterable of integers, return a comma-separated string where
    consecutive runs are represented as "start-end" and single values are
    represented as the value itself.

    Parameters
    ----------
    nums : iterable
        Iterable of integers (may be unsorted and may contain duplicates).

    Returns
    -------
    str
        Comma-separated representation of ranges, e.g. "1-3,5,7-9".

    Notes
    -----
    - An empty input raises an IndexError (matching previous behavior when
      called with an empty list would have raised).
    """
    nums_sorted = sorted(set(nums))
    if not nums_sorted:
        raise IndexError("compress_ranges() arg is an empty sequence")

    parts = []
    start = prev = nums_sorted[0]

    for x in nums_sorted[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x

    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


# def lorch_window(Q, Qmax):
#     """
#     Calculates the Lorch window function to minimize termination ripples.
#     L(Q) = sin(pi*Q/Qmax) / (pi*Q/Qmax)
#     """
#     # Avoid division by zero at Q=0
#     x = np.pi * Q / Qmax
#     L = np.ones_like(Q)
#     # Get non-zero elements
#     nz = (Q != 0)
#     L[nz] = np.sin(x[nz]) / x[nz]
#     return L

# def compute_G_of_r(Q, S_of_Q, r_min=0.0, r_max=20.0, dr=0.01, use_lorch=True):
#     """
#     Computes the Pair Distribution Function G(r) from the structure factor S(Q).

#     Args:
#         Q (np.ndarray): 1D array of momentum transfer values (Å^-1). Must be non-negative and increasing.
#         S_of_Q (np.ndarray): Structure factor S(Q), same shape as Q.
#         r_min (float): Minimum r value for the output grid (Å).
#         r_max (float): Maximum r value for the output grid (Å).
#         dr (float): Step size for the r grid (Å).
#         use_lorch (bool): If True, applies the Lorch window to reduce termination ripples.

#     Returns:
#         tuple: (r, G) where r is the radial distance array and G is the PDF.
#     """
#     # Ensure Q is a sorted numpy array
#     order = np.argsort(Q)
#     Q = np.asarray(Q)[order]
#     S = np.asarray(S_of_Q)[order]

#     # Handle the case where Q does not start at 0
#     if Q[0] > 0:
#         # Linear extrapolation to find S(Q=0)
#         S0 = S[0] + (S[1] - S[0]) * (0 - Q[0]) / (Q[1] - Q[0])
#         Q = np.concatenate(([0.0], Q))
#         S = np.concatenate(([S0], S))

#     Qmax = Q.max()
#     # This is the reduced structure factor, F(Q)
#     FQ = Q * (S - 1.0)

#     # Apply the window function to smooth the cutoff at Qmax
#     if use_lorch:
#         L = lorch_window(Q, Qmax)
#         FQ = FQ * L

#     # Set up the real-space grid
#     r = np.arange(r_min, r_max + dr, dr)
    
#     # Vectorized numerical integration (Sine Fourier Transform)
#     # Using broadcasting to create a (nQ, nr) matrix for the integrand
#     QR = np.outer(Q, r)
#     integrand = FQ[:, None] * np.sin(QR)
    
#     # Trapezoidal rule integration along the Q-axis (axis=0)
#     G = (2.0 / np.pi) * np.trapz(integrand, Q, axis=0)

#     return r, G