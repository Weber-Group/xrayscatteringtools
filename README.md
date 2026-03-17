# X-Ray Scattering Tools

[![Tests](https://github.com/Weber-Group/xrayscatteringtools/actions/workflows/tests.yml/badge.svg)](https://github.com/Weber-Group/xrayscatteringtools/actions/workflows/tests.yml)

`xrayscatteringtools` is a Python library for gas-phase X-ray scattering data analysis. It provides tools for data I/O, visualization, detector calibration, masking, and theoretical modeling. It is designed to work alongside the [CXI-Template](https://github.com/Weber-Group/CXI-Template) repository for analyzing data from the Coherent X-ray Imaging (CXI) endstation at the [Linac Coherent Light Source (LCLS)](https://lcls.slac.stanford.edu/).

## Features

- **Data I/O** — Read and write `.xyz` and `.mol` molecular geometry files, combine multi-run HDF5 experimental data, and manage YAML-based experiment configurations.
- **Visualization** — Plot Jungfrau 4M detector images with proper tiled geometry.
- **Calibration** — Geometry calibration, interactive pixel masking, per-material scattering corrections, and timetool arrival-time calibration.
- **Theoretical Modeling** — Compute elastic and inelastic scattering patterns via the Independent Atom Model (IAM), load _ab initio_ scattering data and optimized molecular geometries.
- **Utilities** — Unit conversions (keV, Å, a.u., _q_, _θ_), momentum-transfer maps, azimuthal binning, molecular transformations, and element lookups.

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

### `xrayscatteringtools.utils`
General-purpose utilities:
- **Unit conversions** — `keV2Angstroms`, `Angstroms2keV`, `au2invAngstroms`, `invAngstroms2au`, `q2theta`, `theta2q`.
- **Detector geometry** — `compute_q_map`, `azimuthalBinning`.
- **Molecular transforms** — `translate_molecule`, `rotate_molecule`.
- **Element lookups** — `element_symbol_to_number`, `element_number_to_symbol`.
- **Other** — `compress_ranges`, `enable_underscore_cleanup`, `J4M` (lazy-loaded Jungfrau 4M geometry).

### `xrayscatteringtools.calib`
Calibration and correction tools:
- **`geometry_calibration`** — Fit beam center and detector distance via azimuthally-averaged scattering patterns (`run_geometry_calibration`, `thompson_correction`, `geometry_correction`).
- **`masking`** — `MaskMaker` class for step-by-step pixel masking: dark, background, polygon, and per-_q_-ring sample masks.
- **`scattering_corrections`** — Per-material correction factors (Si, Al, Be, Kapton HN, sample cell), attenuation-length lookups, and `J4M_efficiency`.
- **`timetool_calibration`** — `fast_erf_fit` for edge detection, `apply_timetool_correction` for per-shot arrival-time correction, and `add_calibration_to_yaml` for persisting calibration parameters.

### `xrayscatteringtools.theory`
Theoretical scattering models:
- **`iam`** — Independent Atom Model: isotropic and oriented elastic/inelastic scattering patterns (`iam_elastic_pattern`, `iam_inelastic_pattern`, `iam_total_pattern`, and their `_oriented` variants), plus `iam_compton_spectrum`.
- **`geometries`** — Lazy-loaded optimized molecular geometries from bundled HDF5 data (accessed as module-level attributes, e.g. `geometries.SF6`).
- **`patterns`** — Lazy-loaded _ab initio_ scattering patterns from bundled HDF5 data (accessed as module-level attributes, e.g. `patterns.SF6__HF__aug_cc_pVDZ`).

---

## Testing

The test suite uses [pytest](https://docs.pytest.org/) and covers all modules (360+ tests):
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