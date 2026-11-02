from types import SimpleNamespace
import h5py
import pathlib
from ..utils import invAngstroms2au, au2invAngstroms

# Generally the import names go like 'molecule_method_basisset' Put all available names in __all__. These should match the .h5 files in the data directory.
__all__ = [
    'SF6__CCSD__aug_cc_pVDZ',
    'SF6__MP2__aug_cc_pVDZ',
    'SF6__HF__aug_cc_pVDZ',
    'Ne__CCSD_X2C__aug_cc_pVDZ_DK',
    'Ne__CCSD_X2C__aug_cc_pVTZ_DK',
    'Ne__CCSD_X2C__aug_cc_pVQZ_DK',
    'SF6__HF__cc_pVTZ',
    'SF6__CCSD_FC__aug_cc_pVDZ',
    'SF6__CCSD_FC__aug_cc_pVTZ',
    'NBD__HF__cc_pVTZ',
    'NBD__CCSD_FC__aug_cc_pVDZ',
    'QC__HF__cc_pVTZ',
    'QC__CCSD_FC__aug_cc_pVDZ',
    ]
def __dir__():
    # Tab completion for IPython
    return sorted(__all__)


_data_path = pathlib.Path(__file__).parent / "data/patterns"

# Optional I_q components a pattern file may carry, with their docstring descriptions.
_COMPONENTS = {
    "I_q_elastic": "Elastic scattering intensity values corresponding to `q`.",
    "I_q_inelastic": "Inelastic scattering intensity values corresponding to `q`.",
    "I_q_hartree": "Hartree (direct) part of the two-electron term, built from the one-electron density alone. Equal to `I_q_elastic`.",
    "I_q_exchange": "Exchange part of the two-electron term.",
    "I_q_cumulant": "Cumulant (electron correlation) part of the two-electron term. Absent for HF, where it is zero.",
}
# Provenance for files made before the 'source' attribute existed
_DEFAULT_SOURCE = "Molpro & PyXSCAT Library. https://github.com/AMC-dyn/PyXSCAT"

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
        A SimpleNamespace object with attributes: q, I_q, molecule, method, basis_set, n_electrons,
        and optionally any of the I_q components in _COMPONENTS, notes, source, and metadata if present in the file.
    """

    attrs = dict(
        q=f["q"][:],
        I_q=f["I_q"][:],
        molecule=f.attrs["molecule"],
        method=f.attrs["method"],
        basis_set=f.attrs["basis_set"],
        n_electrons=f.attrs["n_electrons"],
    )
    for name in _COMPONENTS:
        if name in f:
            attrs[name] = f[name][:]
    for name in ("notes", "source", "metadata"):
        if name in f.attrs:
            attrs[name] = f.attrs[name]
    return SimpleNamespace(**attrs)

def _make_default_docstring(obj):
    """
    Generate a default docstring for a data object if created by data\\patterns\\convert_ab_initio_to_h5.py or data\\patterns\\convert_thistle_to_h5.py

    Parameters
    ----------
    obj : SimpleNamespace
        The data object with attributes: q, I_q, molecule, method, basis_set, n_electrons, and any optional attributes from _make_default_obj.

    Returns
    -------
    doc : str
        The generated docstring.
    """

    components_doc = ""
    for name, description in _COMPONENTS.items():
        if hasattr(obj, name):
            components_doc += f"""    {name} : ndarray of shape ({len(getattr(obj, name))},)
        {description}
"""

    decomposition_doc = ""
    if hasattr(obj, "I_q_hartree"):
        decomposition_doc = "I_q = n_electrons + I_q_hartree + I_q_exchange + I_q_cumulant, and I_q = I_q_elastic + I_q_inelastic."

    notes_doc = ""
    if hasattr(obj, "notes"):
        notes_doc = obj.notes

    doc = f"""
    Ab initio {obj.molecule} scattering data at the {obj.method}/{obj.basis_set} level of theory.

    Attributes
    ----------
    q : ndarray of shape ({len(obj.q)},)
        Momentum transfer values (q) in inverse angstroms. Ranges from {obj.q[0]} to {obj.q[-1]} Å⁻¹. ({invAngstroms2au(obj.q[0])} to {invAngstroms2au(obj.q[-1])} a.u.).
    I_q : ndarray of shape ({len(obj.I_q)},)
        Total scattering intensity values corresponding to `q`.
{components_doc}    molecule : str
        The molecular formula, or an abbreviation for isomers (e.g. NBD, QC).
    method : str
        The method used for the calculation.
    basis_set : str
        The basis set used for the calculation.
    n_electrons : int
        The number of electrons in the molecule.

    Notes
    -----
    Calculated using {getattr(obj, "source", _DEFAULT_SOURCE)}
    Inverse angstroms to atomic units conversion factor used: {au2invAngstroms(1.0)} Å⁻¹/a.u.
    {decomposition_doc}
    {notes_doc}
    """
    
    return doc