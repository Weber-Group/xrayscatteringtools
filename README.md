# xrayscatteringtools

## A python library for the analysis of data from the CXI endstation at the LCLS. 

### List of all methods:
All methods have full docstrings in the NumPy docstring standard.
Proper namespaces have yet to be defined. Some of these functions can remain internal, and the more useful ones can be defined at the surface level.
* calib
  * geometry_calibration
    - run_geometry_calibration()
    - model()
    - thompson_correction()
    - geometry_correction()
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
    - J4M_efficiency()
* theory
  * iam
    - iam_elastic_pattern()
    - iam_inelastic_pattern()
    - iam_total_pattern()
    - iam_compton_spectrum()
  * patterns
    - sf6()
* io
  - combineRuns()
  - get_tree()
  - is_leaf()
  - get_leaves()
  - runNumToString()
  - read_xyz()
* plotting
  - plot_jungfrau()
  - compute_pixel_edges()
* utils
  - enable_underscore_cleanup()
  - azimuthalBinning()
  - keV2Angstroms()
  - Angstroms2keV()
  - q2theta()
  - theta2q()
  - element_symbol_to_number()
  - element_number_to_symbol()
