from xrayscatteringtools.io import read_xyz
from xrayscatteringtools.utils import element_symbol_to_number
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import pathlib
import xraylib
from scipy.constants import physical_constants

base_path = pathlib.Path(__file__).parent

def iam_elastic_pattern(xyzfile, q_arr):
    """
    Compute the elastic (coherent) X-ray scattering intensity (Debye scattering) 
    for a molecule or atomic cluster.

    Parameters
    ----------
    xyzfile : str or Path
        Path to an XYZ file containing the atomic coordinates and element symbols.
    q_arr : array_like
        Array of momentum transfer values (q) at which to evaluate the scattering intensity.

    Returns
    -------
    debye_image : ndarray
        Elastic scattering intensity evaluated at each q in `q_arr`.

    Notes
    -----
    - Uses atomic scattering factors loaded from 'Scattering_Factors.npy'.
    - Includes both atomic self-scattering and molecular interference terms.
    - The molecular interference term is calculated using the Debye formula with np.sinc.
    """
    num_atoms, _, atoms, coords = read_xyz(xyzfile) # Load the data
    coords = np.array(coords)  # Ensure coords is a NumPy array for advanced indexing
    atomic_numbers = [element_symbol_to_number(atom) for atom in atoms] # Get atomic numbers using mendeleev
    scattering_factors_coeffs = np.load(base_path / 'Scattering_Factors.npy', allow_pickle=True)
    scattering_factors = np.zeros((num_atoms, len(q_arr))) # Preallocation for the structure factor array
    q4pi = q_arr / (4 * np.pi)
    for i, atom in enumerate(atomic_numbers): # Loop all atoms
        factor_coeff = scattering_factors_coeffs[atom-1] # Grab the factor coefficients for that element, -1 for zero-based indexing
        # Calculate atomic scattering factor for each q in q_arr
        scattering_factors[i,:] = (
            factor_coeff[0] * np.exp(-factor_coeff[4] * q4pi ** 2) +
            factor_coeff[1] * np.exp(-factor_coeff[5] * q4pi ** 2) +
            factor_coeff[2] * np.exp(-factor_coeff[6] * q4pi ** 2) +
            factor_coeff[3] * np.exp(-factor_coeff[7] * q4pi ** 2) +
            factor_coeff[8]
        )
    # Compute all pairwise distance vectors between atoms
    r_vectors = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # shape: (num_atoms, num_atoms, 3)
    distances = np.linalg.norm(r_vectors, axis=2)  # shape: (num_atoms, num_atoms)

    # Atomic contribution (self-scattering)
    atomic_contribution = np.sum(scattering_factors**2, axis=0)
    elastic_pattern = np.copy(atomic_contribution)

    # Molecular contribution (interference terms)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            a = scattering_factors[i, :]
            b = scattering_factors[j, :]
            r_ij = distances[i, j]
            # np.sinc(x) = sin(pi*x)/(pi*x), so argument is q*r/pi
            molecular_contribution = 2 * a * b * np.sinc(q_arr * r_ij / np.pi)
            elastic_pattern += molecular_contribution

    return elastic_pattern

def iam_inelastic_pattern(xyzfile, q_arr):
    """
    Compute the inelastic (Compton) X-ray scattering intensity for a molecule 
    or atomic cluster.

    Parameters
    ----------
    xyzfile : str or Path
        Path to an XYZ file containing the atomic coordinates and element symbols.
    q_arr : array_like
        Array of momentum transfer values (q) at which to evaluate the scattering intensity.

    Returns
    -------
    inelastic_pattern : ndarray
        Inelastic scattering intensity interpolated at each q in `q_arr`.

    Notes
    -----
    - Uses Compton scattering factors loaded from 'Compton_Factors.npy'.
    - Interpolation is performed using `InterpolatedUnivariateSpline` to return values at the requested q points.
    - The q grid in the Compton factors is fixed; changing it requires updating the array.
    """
    num_atoms, _, atoms, coords = read_xyz(xyzfile) # Load the data
    atomic_numbers = [element_symbol_to_number(atom) for atom in atoms] # Get atomic numbers using mendeleev
    compton_factors = np.load(base_path / 'Compton_Factors.npy') # Load the data
    q_inelastic = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.5, 2., 8., 15.]) * 4*np.pi #These go with the Compton factors array- don't change them unless Compton Array changes.
    inelastic_scattering = np.zeros_like(q_inelastic)
    for atom in atomic_numbers: # Loop through all the atoms
        inelastic_scattering += compton_factors[atom-1,:] # Sum up the contribution
    spline_interp = InterpolatedUnivariateSpline(q_inelastic, inelastic_scattering) # Interpolate the inelastic scattering
    return spline_interp(q_arr) # Return the interpolated inelastic scattering to the desired q values

def iam_total_pattern(xyzfile, q_arr):
    """
    Compute the total X-ray scattering intensity (elastic + inelastic) 
    for a molecule or atomic cluster.

    Parameters
    ----------
    xyzfile : str or Path
        Path to an XYZ file containing the atomic coordinates and element symbols.
    q_arr : array_like
        Array of momentum transfer values (q) at which to evaluate the scattering intensity.

    Returns
    -------
    total_pattern : ndarray
        Total scattering intensity (elastic + inelastic) evaluated at each q in `q_arr`.

    Notes
    -----
    - Combines the outputs of `iam_elastic_pattern` and `iam_inelastic_pattern`.
    - Useful for simulating the full scattering signal from a molecular system.
    """
    return iam_elastic_pattern(xyzfile, q_arr) + iam_inelastic_pattern(xyzfile, q_arr)

def iam_compton_spectrum(formula, theta, EI_keV, EF_keV_array, pz_au_grid=np.linspace(0,100,2000)):
    """
    Compute the Compton scattering spectrum for a molecule or atomic cluster.

    Parameters
    ----------
    formula : str
        Chemical formula of the molecule (e.g., "H2O").
    theta : float or array_like
        Scattering angles in radians. This is the full angle, equal to 2theta in literature.
    EI_keV : float
        Initial photon energy in kilo-electron volts (keV).
    EF_keV_array : array_like
        Array of final photon energies in kilo-electron volts (keV).
    pz_au_grid : array_like, optional
        Array of momentum transfer values at which to evaluate the scattering intensity.
        Default is np.linspace(0,100,2000).

    Returns
    -------
    compton_spectrum : ndarray
        If theta is scalar: shape (len(EF_keV_array),)
        If theta is array: shape (len(theta), len(EF_keV_array))
    """
    # Ensure inputs are arrays
    theta = np.atleast_1d(theta)
    EF_keV_array = np.atleast_1d(EF_keV_array)

    system = xraylib.CompoundParser(formula)
    m_e_c2_keV = physical_constants['electron mass energy equivalent in MeV'][0] * 1000  # keV
    c = physical_constants['speed of light in vacuum'][0]
    p_au = physical_constants['atomic unit of momentum'][0]
    eV2J = physical_constants['electron volt-joule relationship'][0]

    # Convert energies to Joules
    EI_J = EI_keV * 1e3 * eV2J
    EF_J = EF_keV_array * 1e3 * eV2J
    m_e_c2_J = m_e_c2_keV * 1e3 * eV2J

    # Calculate pz for all theta and EF_keV_array (broadcasting)
    cos_theta = np.cos(theta)[:, None]  # shape (Ntheta,1)
    numerator = (EI_J - EF_J)[None, :] * m_e_c2_J - EI_J * EF_J[None, :] * (1 - cos_theta)
    term_under_sqrt = EI_J**2 + EF_J[None, :]**2 - 2*EI_J*EF_J[None, :]*cos_theta
    term_under_sqrt = np.clip(term_under_sqrt, 0, None)  # avoid negative
    denominator = c * np.sqrt(term_under_sqrt)
    pz_si = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    pz_au = pz_si / p_au  # shape (Ntheta, NE)

    # Prepare result
    compton_spectrum = np.zeros((len(theta), len(EF_keV_array)))

    # Loop over elements because xraylib isn't vectorized
    for Zidx, Z in enumerate(system['Elements']):
        # Initialize contribution for this element
        J_element = np.zeros_like(compton_spectrum)
        nAtoms = system['nAtoms'][Zidx]
        for shell in range(0,31):
            try:
                # Build Compton profile for this shell on the pz grid
                J_partial_pz = [
                    xraylib.ComptonProfile_Partial(Z, shell, float(p)) *
                    xraylib.ElectronConfig(Z, shell)
                    for p in pz_au_grid
                ]
                J_partial_interp = interp1d(
                    pz_au_grid, J_partial_pz,
                    bounds_error=False, fill_value=0.0
                )

                # Energy cutoff for this shell
                shell_liftoff = EI_keV - xraylib.EdgeEnergy(Z, shell)

                # Evaluate profile at all |pz| for every theta/E
                for i_th in range(len(theta)):
                    mask = EF_keV_array > shell_liftoff
                    J_vals = J_partial_interp(np.abs(pz_au[i_th, :]))
                    # Apply liftoff: zero contribution where EF > shell_liftoff
                    J_element[i_th, :] += np.where(mask, 0.0, J_vals)
            except Exception:
                # Skip invalid shells
                continue
        compton_spectrum += nAtoms * J_element

    # If theta was scalar, return 1D array
    if compton_spectrum.shape[0] == 1:
        return compton_spectrum[0]
    return compton_spectrum