from types import SimpleNamespace
import h5py
import pathlib

# Generally the import names go like 'molecule_method_basisset' Put all available names in __all__. These should match the .h5 files in the data directory.
__all__ = [
    'SF6__CCSD_T_DHK__aug_cc_pV5Z_DK'
    ]
def __dir__():
    # Tab completion for IPython
    return sorted(__all__)


_data_path = pathlib.Path(__file__).parent / "data/geometries"

def __getattr__(name):
    if name in __all__:
        # Load from HDF5 only when first accessed
        with h5py.File(f"{_data_path}/{name}.h5", "r") as f:
            obj = _make_default_obj(f)
        obj.__doc__ = _make_default_docstring(obj)
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def _make_default_obj(f):
    """
    Create a SimpleNamespace object with default attributes from an HDF5 file.

    Parameters
    ----------
    f : h5py.File
        An open HDF5 file object containing the datasets and attributes.

    Returns
    -------
    obj : SimpleNamespace
        A SimpleNamespace object with attributes: geometry, molecule, method, basis_set, n_electrons, charge, energy, atoms,  notes.
    """
    return SimpleNamespace(
        geometry=f["geometry"][:],
        molecule=f.attrs["molecule"],
        method=f.attrs["method"],
        basis_set=f.attrs["basis_set"],
        n_electrons=f.attrs["n_electrons"],
        charge=f.attrs["charge"],
        energy=f.attrs["energy"],
        atoms=f.attrs["atoms"],
        notes=f.attrs["notes"]
        )

def _make_default_docstring(obj):
    """
    Generate a default docstring for a geometry object.

    Parameters
    ----------
    obj : SimpleNamespace
        The data object with attributes: geometry, molecule, method, basis_set, n_electrons.

    Returns
    -------
    doc : str
        The generated docstring.
    """

    doc = f"""
    Optimized geometry for {obj.molecule} at the {obj.method}/{obj.basis_set} level of theory.

    Attributes
    ----------
    geometry : Nx3 numpy array
        The geometry data of the molecule, in Angstroms.
    atoms : list of str
        The list of atomic symbols corresponding to the geometry.
    molecule : str
        The molecular formula of the species.
    method : str
        The method used for the calculation.
    basis_set : str
        The basis set used for the calculation.
    n_electrons : int
        The number of electrons in the molecule.
    charge : int
        The net charge of the molecule.
    energy : float
        The total electronic energy of the molecule in Hartree.

    Notes
    -----
    {obj.notes}
    """
    return doc