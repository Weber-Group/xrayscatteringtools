# X-Ray Scattering Tools

[![Tests](https://github.com/Weber-Group/xrayscatteringtools/actions/workflows/tests.yml/badge.svg)](https://github.com/Weber-Group/xrayscatteringtools/actions/workflows/tests.yml)

`xrayscatteringtools` is a Python library for gas-phase X-ray scattering data analysis. It provides tools for data I/O, visualization, detector calibration, masking, and theoretical modeling. It is designed to work alongside the [CXI-Template](https://github.com/Weber-Group/CXI-Template) repository for analyzing data from the Coherent X-ray Imaging (CXI) endstation at the [Linac Coherent Light Source (LCLS)](https://lcls.slac.stanford.edu/).

## Features

- **Data I/O** — Read and write `.xyz` and `.mol` molecular geometry files, combine multi-run HDF5 experimental data, and manage YAML-based experiment configurations.
- **Visualization** — Plot Jungfrau 4M detector images with proper tiled geometry.
- **Calibration** — Geometry calibration, interactive pixel masking, per-material scattering corrections, and timetool arrival-time calibration.
- **Theoretical Modeling** — Compute elastic and inelastic scattering patterns via the Independent Atom Model (IAM), load _ab initio_ scattering data and optimized molecular geometries.
- **Azimuthal Integration** — Momentum-transfer maps and azimuthal binning of detector images, either built-in or through [pyFAI](https://pyfai.readthedocs.io/).
- **Utilities** — Unit conversions (keV, Å, a.u., _q_, _θ_), molecular transformations, and element lookups.

---

## Installation

Install from PyPI:
```bash
pip install xrayscatteringtools
```

Or install from source:
```bash
git clone https://github.com/Weber-Group/xrayscatteringtools.git
cd xrayscatteringtools
pip install .
```

For development (includes pytest, black, mypy, ruff):
```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9

---

## Modules Overview

### `xrayscatteringtools.io`
Data input and output:
- `combineRuns` — Combine data from multiple LCLS experimental runs (HDF5).
- `read_xyz` / `write_xyz` — Read and write `.xyz` molecular geometry files.
- `read_mol` — Parse `.mol` / `.molden` files.
- `get_leaves` — Inspect the dataset tree of an HDF5 file.
- `get_data_paths` — Resolve run-specific data paths from a YAML config.
- `get_config` / `get_config_for_runs` — Load experiment configuration values.

### `xrayscatteringtools.plotting`
Detector visualization:
- `plot_j4m` — Plot a Jungfrau 4M image using the stored pixel geometry.
- `plot_jungfrau` — Plot arbitrary Jungfrau panel data with custom coordinates.
- `compute_pixel_edges` / `edges_from_centers` — Derive pixel-edge arrays for `pcolormesh`.

### `xrayscatteringtools.azimuthal`
Azimuthal integration of detector images (all distances in microns, angles in degrees, energy in keV):
- `compute_q_map` — Per-pixel momentum transfer for a given detector geometry.
- `azimuthalBinning` — Bin an image into _q_ (or _r_) and optional azimuthal sectors, with polarization and solid-angle corrections.
- `create_J4m_integrator` — Build a pyFAI `AzimuthalIntegrator` from the real Jungfrau 4M pixel layout.
- `azimuthalBinning_pyfai` — pyFAI-based alternative to `azimuthalBinning` that takes the same geometry parameters. Pass a pre-built integrator (`ai=`) to reuse it across many images.

### `xrayscatteringtools.utils`
General-purpose utilities:
- **Unit conversions** — `keV2Angstroms`, `Angstroms2keV`, `au2invAngstroms`, `invAngstroms2au`, `q2theta`, `theta2q`.
- **Molecular transforms** — `translate_molecule`, `rotate_molecule`.
- **Element lookups** — `element_symbol_to_number`, `element_number_to_symbol`.
- **Other** — `compress_ranges`, `expand_ranges`, `enable_underscore_cleanup`, `J4M` (lazy-loaded Jungfrau 4M geometry).

### `xrayscatteringtools.calib`
Calibration and correction tools:
- **`geometry_calibration`** — Fit beam center and detector distance to a theory pattern, with an optional flat background (`run_geometry_calibration`, `thompson_correction`, `geometry_correction`).
- **`masking`** — `MaskMaker` class for step-by-step pixel masking: dark, background, polygon, and per-_q_-ring sample masks. The detector's line and T masks are applied from the start, so known-bad pixels never enter the statistics.
- **`scattering_corrections`** — Per-material correction factors (Si, Al, Be, Kapton HN, sample cell), attenuation lengths from xraylib, and `J4M_efficiency`.
- **`timetool_calibration`** — `fast_erf_fit` for edge detection, `apply_timetool_correction` for per-shot arrival-time correction, and `add_calibration_to_yaml` for persisting calibration parameters.

### `xrayscatteringtools.theory`
Theoretical scattering models:
- **`iam`** — Independent Atom Model: isotropic and oriented elastic/inelastic scattering patterns (`iam_elastic_pattern`, `iam_inelastic_pattern`, `iam_total_pattern`, and their `_oriented` variants), `iam_compton_spectrum`, and `iam_inelastic_pattern_from_formula` (Compton intensity from a chemical formula alone, no coordinates needed).
- **`geometries`** — Lazy-loaded optimized molecular geometries from bundled HDF5 data (accessed as module-level attributes, e.g. `geometries.SF6__CCSD_T_DHK__aug_cc_pV5Z_DK`).
- **`patterns`** — Lazy-loaded _ab initio_ scattering patterns from bundled HDF5 data (accessed as module-level attributes named `molecule__method__basis_set`, e.g. `patterns.SF6__HF__aug_cc_pVDZ`). Available: SF6 at the HF, MP2, and CCSD levels (aug-cc-pVDZ), and Ne at the CCSD+X2C level (aug-cc-pVDZ/TZ/QZ+DK). Some patterns carry a `notes` attribute with details of the calculation.

---

## Testing

The test suite uses [pytest](https://docs.pytest.org/) and covers all modules (460+ tests):
```bash
pip install -e ".[dev]"
python -m pytest tests/
```

Tests run automatically via GitHub Actions on every push and pull request to `main`.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This library was created by [David J. Romano](https://david.lizaanddavid.com) by compiling and standardizing code from previous data analysis pipelines. Maintained by the [Weber Research Group](https://sites.brown.edu/weber-lab/) and developed by collaborators to facilitate X-ray scattering research.