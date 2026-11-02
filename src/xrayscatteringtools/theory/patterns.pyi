from types import SimpleNamespace
from typing import Optional

__all__: list[str]

class _PatternData(SimpleNamespace):
    """Type stub for ab initio scattering pattern data."""
    q: any  # ndarray
    I_q: any  # ndarray
    I_q_elastic: Optional[any]  # ndarray, may not be present
    I_q_inelastic: Optional[any]  # ndarray, may not be present
    I_q_hartree: Optional[any]  # ndarray, may not be present
    I_q_exchange: Optional[any]  # ndarray, may not be present
    I_q_cumulant: Optional[any]  # ndarray, may not be present
    molecule: str
    method: str
    basis_set: str
    n_electrons: int
    notes: Optional[any]
    source: Optional[str]
    metadata: Optional[str]  # JSON string

SF6__CCSD__aug_cc_pVDZ: _PatternData
SF6__MP2__aug_cc_pVDZ: _PatternData
SF6__HF__aug_cc_pVDZ: _PatternData
Ne__CCSD_X2C__aug_cc_pVDZ_DK: _PatternData
Ne__CCSD_X2C__aug_cc_pVTZ_DK: _PatternData
Ne__CCSD_X2C__aug_cc_pVQZ_DK: _PatternData
SF6__HF__cc_pVTZ: _PatternData
SF6__CCSD_FC__aug_cc_pVDZ: _PatternData
SF6__CCSD_FC__aug_cc_pVTZ: _PatternData
NBD__HF__cc_pVTZ: _PatternData
NBD__CCSD_FC__aug_cc_pVDZ: _PatternData
QC__HF__cc_pVTZ: _PatternData
QC__CCSD_FC__aug_cc_pVDZ: _PatternData
