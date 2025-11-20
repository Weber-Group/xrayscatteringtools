from types import SimpleNamespace
import h5py
import pathlib
from ..utils import invAngstroms2au, au2invAngstroms

# Generally the import names go like 'molecule_method_basisset' Put all available names in __all__. These should match the .h5 files in the data directory.
__all__ = [
    'SF6__CCSD__aug_cc_pVDZ',
    'SF6__MP2__aug_cc_pVDZ',
    'SF6__HF__aug_cc_pVDZ'
    ]
def __dir__():
    # Tab completion for IPython
    return sorted(__all__)


_data_path = pathlib.Path(__file__).parent / "data/patterns"

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
        A SimpleNamespace object with attributes: q, I_q, I_q_elastic, I_q_inelastic, molecule, method, basis_set, n_electrons.
    """

    q = f["q"][:]     # read dataset into memory
    I_q = f["I_q"][:]
    I_q_elastic = f["I_q_elastic"][:]
    I_q_inelastic = f["I_q_inelastic"][:]
    molecule = f.attrs["molecule"]
    method = f.attrs["method"]
    basis_set = f.attrs["basis_set"]
    n_electrons = f.attrs["n_electrons"]
    return SimpleNamespace(q=q, I_q=I_q, I_q_elastic=I_q_elastic, I_q_inelastic=I_q_inelastic, molecule=molecule, method=method, basis_set=basis_set, n_electrons=n_electrons)

def _make_default_docstring(obj):
    """
    Generate a default docstring for a data object if created by tests\\convert_ab_initio_to_h5.py

    Parameters
    ----------
    obj : SimpleNamespace
        The data object with attributes: q, I_q, I_q_elastic, I_q_inelastic, molecule, method, basis_set, n_electrons.

    Returns
    -------
    doc : str
        The generated docstring.
    """

    doc = f"""
    Ab initio {obj.molecule} scattering data at the {obj.method}/{obj.basis_set} level of theory.

    Attributes
    ----------
    q : ndarray of shape ({len(obj.q)},)
        Momentum transfer values (q) in inverse angstroms. Ranges from {obj.q[0]} to {obj.q[-1]} Å⁻¹. ({invAngstroms2au(obj.q[0])} to {invAngstroms2au(obj.q[-1])} a.u.).
    I_q : ndarray of shape ({len(obj.I_q)},)
        Total scattering intensity values corresponding to `q`.
    I_q_elastic : ndarray of shape ({len(obj.I_q_elastic)},)
        Elastic scattering intensity values corresponding to `q`.
    I_q_inelastic : ndarray of shape ({len(obj.I_q_inelastic)},)
        Inelastic scattering intensity values corresponding to `q`.
    molecule : str
        The molecular formula of the species.
    method : str
        The method used for the calculation.
    basis_set : str
        The basis set used for the calculation.
    n_electrons : int
        The number of electrons in the molecule.

    Notes
    -----
    Calculated using Molpro & PyXSCAT Library. https://github.com/AMC-dyn/PyXSCAT
    Inverse angstroms to atomic units conversion factor used: {au2invAngstroms(1.0)} Å⁻¹/a.u.
    """
    return doc