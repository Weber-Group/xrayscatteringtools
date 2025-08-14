import re
from IPython import get_ipython

def enable_underscore_cleanup():
    """Registers a post-cell hook to delete user-defined _ variables after each cell."""
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

def keV2Angstroms(keV):
    """Convert energy in keV to wavelength in Angstroms."""
    return 12.39841984/keV

def Angstroms2keV(angstroms):
    """Convert wavelength in Angstroms to energy in keV."""
    return 12.39841984/angstroms

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