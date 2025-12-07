# X-Ray Scattering Tools

`xrayscatteringtools` is a Python library designed for advanced X-ray scattering data analysis. It provides tools for data processing, visualization, calibration, and theoretical modeling. Specifically, it works alongside the [CXI-Template](https://github.com/Weber-Group/CXI-Template) repository for analyzing data from the Linac Coherent Light Source Coherent X-ray Imaging hutch.

## Features

- **Data I/O**: Read and write various data formats, including `.xyz` and `.mol` files, and handle experimental data stored in HDF5.
- **Visualization**: Generate plots for detector data.
- **Calibration**: Perform geometry calibration and create masks for X-ray scattering experiments.
- **Theoretical Modeling**: Compute scattering patterns using theoretical models and atomic data.
- **Utilities**: A collection of helper functions for unit conversions, azimuthal binning, and more.

---

## Installation

To install the latest version, use pip:
```bash
pip install xrayscatteringtools
```

or install manually
```bash
git clone https://github.com/Weber-Group/xrayscatteringtools.git
cd xrayscatteringtools
pip install -r requirements.txt
```

---

## Modules Overview

### 1. `xrayscatteringtools.io`
Handles data input and output operations:
- **Key Functions**:
  - `combineRuns`: Combine data from multiple experimental runs.
  - `read_xyz`, `write_xyz`: Read and write `.xyz` files.
  - `get_data_paths`: Retrieve paths to data files.
  - `get_config`: Load configuration files for experiments.

### 2. `xrayscatteringtools.plotting`
Provides tools for visualizing X-ray scattering data:
- **Key Functions**:
  - `plot_j4m`: Plot Jungfrau 4M detector data.

### 3. `xrayscatteringtools.utils`
A collection of utility functions:
- **Key Functions**
  - Unit conversions, azimuthal binning, transformations, ipython ease-of-use.

### 4. `xrayscatteringtools.calib`
Tools for calibration and masking:
- **Submodules**:
  - `geometry_calibration`: Perform geometry calibration using theoretical scattering patterns.
  - `masking`: Create and manage masks for X-ray scattering data.
  - `scattering_corrections`: Compute correction factors for scattering experiments.

### 5. `xrayscatteringtools.theory`
Theoretical modeling of X-ray scattering:
- **Submodules**:
  - `geometries`: Load and manage molecular geometries from HDF5 files.
  - `iam`: Compute elastic X-ray scattering patterns using the Independent Atom Model (IAM).
  - `patterns` : Load and manage _ab initio_ scattering data from HDF5 files.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

This library was created by David J. Romano and developed by collaborators to facilitate X-ray scattering research.
