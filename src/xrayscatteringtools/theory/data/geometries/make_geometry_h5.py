import h5py
import numpy as np

# Naming convention "Molecule__Method__Basis__Extras"
filename = "SF6__CCSD_T_DHK__aug_cc_pV5Z_DK"
molecule = "SF6"
method = "CCSD(T)+DHK"
basis_set = "aug-cc-pV5Z-DK"
n_electrons = 70
energy = -998.01159357
notes = "+DKH means that scalar-relativistic corrections via a tenth-order Douglas-Kroll-Hess Hamiltonian are included. This is also reflected by the -DK in the basis set. Mats Simmermacher sent over on Nov 17, 2025."
atoms = ['S', 'F', 'F', 'F', 'F', 'F', 'F']
charge = 0
geometry = np.array([
    [ 0.0000000000,  0.0000000000,  0.0000000000],
    [ 0.0000000000,  0.0000000000,  1.5600310790],
    [ 0.0000000000,  0.0000000000, -1.5600310790],
    [ 0.0000000000,  1.5600310790,  0.0000000000],
    [ 0.0000000000, -1.5600310790,  0.0000000000],
    [ 1.5600310790,  0.0000000000,  0.0000000000],
    [-1.5600310790,  0.0000000000,  0.0000000000]
    ])

# Creating the file and saving it
with h5py.File(filename + ".h5", "w") as f:

    # Attributes
    f.attrs["molecule"] = molecule
    f.attrs["method"] = method
    f.attrs["basis_set"] = basis_set
    f.attrs["n_electrons"] = n_electrons
    f.attrs["energy"] = energy
    f.attrs["notes"] = notes
    f.attrs["charge"] = charge
    f.attrs["atoms"] = atoms

    # Datasets
    f.create_dataset("geometry", data=geometry)