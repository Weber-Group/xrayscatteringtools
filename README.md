# xrayscatteringtools

## A python library for the analysis of data from the CXI endstation at the LCLS. 

### `calib` submodule pertains to the geometry calibration routine

### `theory` submodule pertains to stored ab initio patterns, iam, or geometries.
### `utils` has a bunch of useful functions in it
### `xrayscatteringtools.io` has anything pertaning to loading or writing data (`combineRuns` is in here)
### `plotting` has anything to do with plotting.

### List of all methods:
All methods have full docstrings in the NumPy docstring standard.
Proper namespaces have yet to be defined. Some of these functions can remain internal, and the more useful ones can be defined at the surface level.
* calib
  * geometry_calibration
    - run_geometry_calibration()
    - model()
    - thompson_correction()
    - geometry_correction()
    - geometry_correction_units()
  * scattering_corrections
    - correction_factor()
    - Si_correction()
    - KaptonHN_correction()
    - Al_correction()
    - Be_correction()
    - cell_correction()
    - Si_attenuation_length()
    - Al_attenuation_length()
    - Be_attenuation_length()
    - KaptonHN_attenuation_length()
    - Zn_attenuation_length()
    - J4M_efficiency()
* theory
  * iam
    - iam_elastic_pattern()
    - iam_inelastic_pattern()
    - iam_total_pattern()
    - iam_compton_spectrum()
    - iam_elastic_pattern_oriented()
    - iam_inelastic_pattern_oriented()
    - iam_total_pattern_oriented()
  * patterns
    - SF6__CCSD__aug_cc_pVDZ
    - SF6__HF__aug_cc_pVDZ
    - SF6__MP2__aug_cc_pVDZ
  * geometries
    - SF6__CCSD_T_DHK__aug_cc_pV5Z_DK 
* io
  - combineRuns()
  - get_tree()
  - is_leaf()
  - get_leaves()
  - get_data_paths()
  - runNumToString()
  - read_xyz()
  - write_xyz()
  - read_mol()
* plotting
  - plot_jungfrau()
  - compute_pixel_edges()
* utils
  - enable_underscore_cleanup()
  - azimuthalBinning()
  - au2invAngstroms()
  - invAngstroms2au()
  - keV2Angstroms()
  - Angstroms2keV()
  - q2theta()
  - theta2q()
  - element_symbol_to_number()
  - element_number_to_symbol()
  - translate_molecule()
  - rotate_molecule()
  - J4M()
  - 
