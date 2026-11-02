import json
import numpy as np
import h5py

# Converts thistle .csv exports into the .h5 format read by xrayscatteringtools.theory.patterns.
# Expected .csv layout:
#   line 1: "# <label>  q in 1/A  {json metadata}"
#   line 2: "q (1/A),total,elastic,inelastic,hartree,exchange[,cumulant]"
#   rest:   comma separated data, q in inverse angstroms
# The total satisfies I_q = n_electrons + hartree + exchange + cumulant (cumulant is zero for HF).
# Paths are relative to the current working directory.

##################### User Set Parameters #####################
# (csv file, molecule, method, basis set, n_electrons, notes)
# h5 file naming convention is molecule__method__basisset.h5
NBD_NOTES = "NBD is norbornadiene (bicyclo[2.2.1]hepta-2,5-diene, C7H8)."
QC_NOTES = "QC is quadricyclane (tetracyclo[3.2.0.0^{2,7}.0^{4,6}]heptane, C7H8), the valence isomer of norbornadiene."
FC_NOTES = "CCSD with frozen core electrons."
conversions = [
    ("nb_hf_cc-pvtz.csv", "NBD", "HF", "cc_pVTZ", 50, NBD_NOTES),
    ("nb_ccsd_aug-cc-pvdz_fc.csv", "NBD", "CCSD_FC", "aug_cc_pVDZ", 50, f"{NBD_NOTES} {FC_NOTES}"),
    ("qd_hf_cc-pvtz.csv", "QC", "HF", "cc_pVTZ", 50, QC_NOTES),
    ("qd_ccsd_aug-cc-pvdz_fc.csv", "QC", "CCSD_FC", "aug_cc_pVDZ", 50, f"{QC_NOTES} {FC_NOTES}"),
    ("sf6_hf_cc-pvtz.csv", "SF6", "HF", "cc_pVTZ", 70, ""),
    ("sf6_ccsd_aug-cc-pvdz_fc.csv", "SF6", "CCSD_FC", "aug_cc_pVDZ", 70, FC_NOTES),
    ("sf6_ccsd_aug-cc-pvtz_fc.csv", "SF6", "CCSD_FC", "aug_cc_pVTZ", 70, FC_NOTES),
]
##################### End User Set Parameters #####################

# csv column -> h5 dataset
column_map = {
    "q (1/A)": "q",
    "total": "I_q",
    "elastic": "I_q_elastic",
    "inelastic": "I_q_inelastic",
    "hartree": "I_q_hartree",
    "exchange": "I_q_exchange",
    "cumulant": "I_q_cumulant",
}

for csv_file, molecule, method, basis_set, n_electrons, notes in conversions:
    with open(csv_file) as f:
        header = f.readline()
        columns = f.readline().strip().split(",")
    metadata = json.loads(header[header.index("{"):])
    if metadata["q_unit"] != "1/A":
        raise ValueError(f"{csv_file}: expected q in 1/A, got {metadata['q_unit']}")
    data = np.loadtxt(csv_file, delimiter=",", skiprows=2)
    datasets = {column_map[c]: data[:, i] for i, c in enumerate(columns)}
    # Forward scattering is n_electrons**2, catches a wrong electron count
    if not np.isclose(datasets["I_q"][0], n_electrons**2, rtol=1e-6) or datasets["q"][0] != 0:
        raise ValueError(f"{csv_file}: I(q=0) = {datasets['I_q'][0]}, expected {n_electrons**2}")

    h5_filename = f"{molecule}__{method}__{basis_set}.h5"
    with h5py.File(h5_filename, "w") as f:
        for name, values in datasets.items():
            f.create_dataset(name, data=values)
        # Add metadata as attributes
        f.attrs["molecule"] = molecule
        f.attrs["method"] = method
        f.attrs["basis_set"] = basis_set
        f.attrs["n_electrons"] = n_electrons
        f.attrs["source"] = f"{metadata['from']['source']} via thistle {metadata['thistle']} ({metadata['kind']}, computed {metadata['created']})"
        f.attrs["metadata"] = json.dumps(metadata)
        if notes:
            f.attrs["notes"] = notes
    print(f"{csv_file} -> {h5_filename}")
