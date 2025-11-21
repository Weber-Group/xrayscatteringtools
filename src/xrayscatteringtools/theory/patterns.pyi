from types import SimpleNamespace

__all__: list[str]

class _PatternData(SimpleNamespace):
    """Type stub for ab initio scattering pattern data."""
    q: any  # ndarray
    I_q: any  # ndarray
    I_q_elastic: any  # ndarray
    I_q_inelastic: any  # ndarray
    molecule: str
    method: str
    basis_set: str
    n_electrons: int

SF6__CCSD__aug_cc_pVDZ: _PatternData
SF6__MP2__aug_cc_pVDZ: _PatternData
SF6__HF__aug_cc_pVDZ: _PatternData
