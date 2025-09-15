import numpy as np
from types import SimpleNamespace
import h5py
import pathlib

__all__ = ['sf6'] # So that from 'xrayscatteringtools.theory.patterns import *' works
data_path = pathlib.Path(__file__).parent / "data"


def __getattr__(name):
    if name == "sf6":
        # Load from HDF5 only when first accessed
        with h5py.File(f"{data_path}/sf6.h5", "r") as f:
            q = f["q"][:]     # read dataset into memory
            I_q = f["I_q"][:]
            I_q_elastic = f["I_q_elastic"][:]
            I_q_inelastic = f["I_q_inelastic"][:]
        obj = SimpleNamespace(q=q, I_q=I_q, I_q_elastic=I_q_elastic, I_q_inelastic=I_q_inelastic)
        obj.__doc__ = """
        Ab initio SF6 Scattering Data.
        
        Attributes
        ----------
        q : ndarray of shape (250,)
            Momentum transfer values (q) in inverse angstroms.
        I_q : ndarray of shape (250,)
            Scattering intensity values corresponding to `q`.
        I_q_elastic : ndarray of shape (250,)
            Elastic scattering intensity values corresponding to `q`.
        I_q_inelastic : ndarray of shape (250,)
            Inelastic scattering intensity values corresponding to `q`.

        Notes
        -----
        The total scattering data was received from Andres M. Carrascosa on Aug 26, 2025. The elastic calculation was received on Sept 15, 2025.
        Andres notes that both were calculated at the CCSD/AVDZ level of theory. The conversion from a.u. to 1/Angstrom was done using the factor 1.8897261246257702.
        """
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    # Tab completion for IPython
    return sorted(__all__)