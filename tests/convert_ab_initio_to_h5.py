import numpy as np
import h5py
from xrayscatteringtools.utils import au2invAngstroms

# IMPORTANT: This script assumes that the .dat files are in the current working directory.
# If they are located elsewhere, please adjust the file paths accordingly.
# Also, the elastic and total data should be on the same q grid.

##################### User Set Parameters #####################
dat_file_elastic = "sf6_elastic_HF.dat"
dat_file_total = "sf6_total_HF.dat"

n_electrons = 70  # Number of electrons in the molecule

# h5 file naming convention is molecule_method_basisset.h5
molecule = "SF6"
method = "HF"
basis_set = "aug_cc_pVDZ"
##################### End User Set Parameters #####################
h5_filename = f"{molecule}_{method}_{basis_set}.h5"

# Current conversion from a.u. to 1/Angstrom
conversion_factor = au2invAngstroms(1.0)


elastic_data = np.loadtxt(dat_file_elastic)
total_data = np.loadtxt(dat_file_total)
if not np.allclose(elastic_data[:,0], total_data[:,0]):
    raise ValueError("q grids in elastic and total data do not match!")

q = total_data[:, 0] * conversion_factor  # q in 1/Angstrom
I_q = total_data[:, 1] + n_electrons   # Total scattering intensity
I_q_elastic = elastic_data[:, 1]  # Elastic scattering intensity
I_q_inelastic = I_q - I_q_elastic  # Inelastic scattering intensity

with h5py.File(h5_filename, "w") as f:
    f.create_dataset("q", data=q)
    f.create_dataset("I_q", data=I_q)
    f.create_dataset("I_q_elastic", data=I_q_elastic)
    f.create_dataset("I_q_inelastic", data=I_q_inelastic)
    # Add metadata as attributes
    f.attrs["molecule"] = molecule
    f.attrs["method"] = method
    f.attrs["basis_set"] = basis_set
    f.attrs["n_electrons"] = n_electrons