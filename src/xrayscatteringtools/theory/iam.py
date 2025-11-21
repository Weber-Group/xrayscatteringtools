from ..io import read_xyz, read_mol
from ..utils import element_symbol_to_number
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import pathlib
import xraylib
from scipy.constants import physical_constants
import types

base_path = pathlib.Path(__file__).parent

def _iam_loader(system):
    """
    Helper function to extract atomic data from various input types.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.

    Returns
    -------
    num_atoms : int
        Number of atoms in the system.
    atoms : list of str
        List of atomic symbols.
    coords : ndarray
        Array of atomic coordinates with shape (num_atoms, 3).
    """
    if isinstance(system,str):
        if system.endswith('.xyz'):
            num_atoms, _, atoms, coords = read_xyz(system) # Load the data
        elif system.endswith('.mol'):
            _, _, _, num_atoms, _, atoms, coords, _, _, = read_mol(system)
    elif isinstance(system,types.SimpleNamespace):
        num_atoms = len(system.atoms)
        coords = system.geometry
        atoms = system.atoms
    return num_atoms, atoms, coords        

def iam_elastic_pattern(system, q_arr):
    """
    Compute the elastic (coherent) X-ray scattering intensity (Debye scattering) 
    for a molecule or atomic cluster.

    Parameters
    ----------
    system : str or Path to xyz file or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
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
    num_atoms, atoms, coords = _iam_loader(system)
    
    coords = np.array(coords)  # Ensure coords is a NumPy array for advanced indexing
    atomic_numbers = [element_symbol_to_number(atom) for atom in atoms] # Get atomic numbers using mendeleev
    scattering_factors_coeffs = np.load(base_path / 'data/IAM/Scattering_Factors.npy', allow_pickle=True)
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

def iam_inelastic_pattern(system, q_arr):
    """
    Compute the inelastic (Compton) X-ray scattering intensity for a molecule 
    or atomic cluster.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
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
    # Getting data from the source
    num_atoms, atoms, coords = _iam_loader(system)
    atomic_numbers = [element_symbol_to_number(atom) for atom in atoms] # Get atomic numbers using mendeleev
    compton_factors = np.load(base_path / 'data/IAM/Compton_Factors.npy') # Load the data
    q_inelastic = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.5, 2., 8., 15.]) * 4*np.pi #These go with the Compton factors array- don't change them unless Compton Array changes.
    inelastic_scattering = np.zeros_like(q_inelastic)
    for atom in atomic_numbers: # Loop through all the atoms
        inelastic_scattering += compton_factors[atom-1,:] # Sum up the contribution
    spline_interp = InterpolatedUnivariateSpline(q_inelastic, inelastic_scattering) # Interpolate the inelastic scattering
    return spline_interp(q_arr) # Return the interpolated inelastic scattering to the desired q values

def iam_total_pattern(system, q_arr):
    """
    Compute the total X-ray scattering intensity (elastic + inelastic) 
    for a molecule or atomic cluster.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
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
    return iam_elastic_pattern(system, q_arr) + iam_inelastic_pattern(system, q_arr)

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

def iam_elastic_pattern_oriented(system, q_arr, phi_arr):
    """
    Computes the elastic X-ray scattering pattern for a fixed-orientation molecule
    on a flat, 2D polar grid defined by q (magnitude) and phi (azimuthal angle).

    This corresponds to a planar slice through the 3D scattering volume, typically
    the qx-qy plane.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
    q_arr : array_like
        1D array of momentum transfer magnitudes (q) in inverse Angstroms. These
        are the radial coordinates for the output grid.
    phi_arr : array_like
        1D array of azimuthal angles (phi) in radians [0, 2*pi]. These are the
        angular coordinates for the output grid.

    Returns
    -------
    oriented_pattern : ndarray
        2D array of the elastic scattering intensity, with shape
        (len(q_arr), len(phi_arr)).
    """
    # 1. Read atomic data
    num_atoms, atoms, coords = _iam_loader(system)
    coords = np.array(coords)
    atomic_numbers = [element_symbol_to_number(atom) for atom in atoms]
    scattering_factors_coeffs = np.load(base_path / 'data/IAM/Scattering_Factors.npy', allow_pickle=True)

    # 2. Create a polar grid of q-vectors in the qx-qy plane
    q_grid, phi_grid = np.meshgrid(q_arr, phi_arr, indexing='ij')

    # 3. Calculate q-dependent atomic scattering factors F(q)
    # F(q) depends only on the magnitude of q, which is our radial coordinate.
    scattering_factors = np.zeros((num_atoms, len(q_arr), len(phi_arr)))
    q4pi_sq_grid = (q_grid / (4 * np.pi))**2
    for i, atom_num in enumerate(atomic_numbers):
        fc = scattering_factors_coeffs[atom_num - 1]
        scattering_factors[i, :, :] = (
            fc[0] * np.exp(-fc[4] * q4pi_sq_grid) + fc[1] * np.exp(-fc[5] * q4pi_sq_grid) +
            fc[2] * np.exp(-fc[6] * q4pi_sq_grid) + fc[3] * np.exp(-fc[7] * q4pi_sq_grid) +
            fc[8]
        )

    # 4. Calculate the total scattering amplitude
    # Define the q-vectors for our planar slice (qz = 0)
    qx = q_grid * np.cos(phi_grid)
    qy = q_grid * np.sin(phi_grid)
    qz = np.zeros_like(q_grid)
    q_vectors_grid = np.stack([qx, qy, qz], axis=-1)

    # Calculate dot products (q . R_k)
    dot_products = np.tensordot(coords, q_vectors_grid, axes=([1], [2]))

    # Calculate complex atomic amplitudes: A_k = F_k(q) * exp(i * q . R_k)
    atomic_amplitudes = scattering_factors * np.exp(1j * dot_products)

    # Sum over all atoms
    total_amplitude = np.sum(atomic_amplitudes, axis=0)

    # 5. Intensity is the squared magnitude of the total amplitude
    oriented_pattern = np.abs(total_amplitude)**2

    return oriented_pattern

def iam_inelastic_pattern_oriented(system, q_arr, phi_arr):
    """
    Computes the inelastic (Compton) X-ray scattering pattern for a fixed-orientation molecule
    on a flat, 2D polar grid defined by q (magnitude) and phi (azimuthal angle).

    This corresponds to a planar slice through the 3D scattering volume, typically
    the qx-qy plane.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
    q_arr : array_like
        1D array of momentum transfer magnitudes (q) in inverse Angstroms. These
        are the radial coordinates for the output grid.
    phi_arr : array_like
        1D array of azimuthal angles (phi) in radians [0, 2*pi]. These are the
        angular coordinates for the output grid.

    Returns
    -------
    oriented_pattern : ndarray
        2D array of the inelastic scattering intensity, with shape
        (len(q_arr), len(phi_arr)).
    """

    inelastic_pattern = iam_inelastic_pattern(system, q_arr)  # Get the 1D inelastic pattern
    oriented_pattern = np.tile(inelastic_pattern[:, np.newaxis], (1, len(phi_arr)))  # Expand to 2D
    return oriented_pattern

def iam_total_pattern_oriented(system, q_arr, phi_arr):
    """
    Computes the total X-ray scattering pattern (elastic + inelastic) for a fixed-orientation molecule
    on a flat, 2D polar grid defined by q (magnitude) and phi (azimuthal angle).

    This corresponds to a planar slice through the 3D scattering volume, typically
    the qx-qy plane.

    Parameters
    ----------
    system : str or xrayscatteringtools.theory.geometries object
        Path to an XYZ or MOL file containing the atomic coordinates and element symbols, or xrayscatteringtools.theory.geometries object.
    q_arr : array_like
        1D array of momentum transfer magnitudes (q) in inverse Angstroms. These
        are the radial coordinates for the output grid.
    phi_arr : array_like
        1D array of azimuthal angles (phi) in radians [0, 2*pi]. These are the
        angular coordinates for the output grid.
    Returns
    -------
    oriented_pattern : ndarray
        2D array of the total scattering intensity (elastic + inelastic), with shape
        (len(q_arr), len(phi_arr)).
    """
    return iam_elastic_pattern_oriented(system, q_arr, phi_arr) + iam_inelastic_pattern_oriented(system, q_arr, phi_arr)

