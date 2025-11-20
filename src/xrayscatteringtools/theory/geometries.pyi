from types import SimpleNamespace

__all__: list[str]

class _GeometryData(SimpleNamespace):
    """Type stub for ab initio geometry data."""
    geometry: any  # ndarray
    molecule: str
    method: str
    basis_set: str
    n_electrons: int
    charge: int
    energy: float
    atoms: list[str]
    notes: str

SF6__CCSD_T_DHK__aug_cc_pV5Z_DK: _GeometryData
