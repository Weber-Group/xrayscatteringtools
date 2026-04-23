from scipy.interpolate import InterpolatedUnivariateSpline
from ..utils import q2theta, theta2q
import numpy as np
import pathlib
_data_path = pathlib.Path(__file__).parent / "data"
import h5py


def _load_attenuation_table(material_key):
    """Load attenuation length table from HDF5 and return (E_values, length) arrays."""
    with h5py.File(_data_path / "attenuation_lengths.h5", 'r') as f:
        E_values = f['E_values'][:]
        length = f[f'{material_key}_length'][:]
    return E_values, length

def correction_factor(
    q_arr, 
    keV, 
    L=2.4e-3, 
    tSi=318.5e-6, 
    tK=8e-6, 
    tAl=4.5e-6, 
    tBe=100e-6, 
    rBe=250e-6, 
    tP=125e-6, 
    rP=125e-6
):
    """
    Calculate the total X-ray scattering correction factor for a given setup.
    
    The total correction factor accounts for absorption and scattering 
    from different materials in the beam path, including Si, Kapton, 
    Al, Be, and the gas cell itself.

    Parameters
    ----------
    qbins : array-like
        Array of momentum transfer (q) values in Å⁻¹ or equivalent units.
    keV : float
        Photon energy in keV.
    L : float, optional
        Gas cell length in meters. Default is 2.4e-3 m.
    tSi : float, optional
        Thickness of silicon in meters. Default is 318.5e-6 m.
    tK : float, optional
        Thickness of Kapton in meters. Default is 8e-6 m.
    tAl : float, optional
        Thickness of aluminum in meters. Default is 4.5e-6 m.
    tBe : float, optional
        Thickness of beryllium in meters. Default is 100e-6 m.
    rBe : float, optional
        Radius of the Be window in meters. Default is 250e-6 m.
    tP : float, optional
        Thickness of the gas cell in meters. Default is 125e-6 m.
    rP : float, optional
        Hole radius of the gas cell platinum pinhole in meters. Default is 125e-6 m.

    Returns
    -------
    numpy.ndarray
        Array of total correction factors corresponding to each q value.
    
    Notes
    -----
    The total correction factor is computed as the product of individual 
    material corrections:
    
        total_correction = Si_correction * KaptonHN_correction * 
                           Al_correction * Be_correction * cell_correction
    """
    q_arr = np.asarray(q_arr, dtype=float)
    if keV <= 0:
        raise ValueError(f"'keV' must be positive, got {keV}.")
    return (
        Si_correction(q_arr, keV, tSi) *
        KaptonHN_correction(q_arr, keV, tK) *
        Al_correction(q_arr, keV, tAl) *
        Be_correction(q_arr, keV, tBe, rBe, L) *
        cell_correction(q_arr, keV, tP, rP, L)
    )

def Si_correction(q_arr, keV, tSi=318.5e-6):
    """
    Calculate the Silicon correction factor.

    Parameters
    ----------
    q_arr : array-like
        Array of momentum transfer (q) values.
    keV : float
        Photon energy in keV.
    tSi : float, optional
        Silicon thickness in meters. Default is 318.5e-6 m.

    Returns
    -------
    numpy.ndarray
        Silicon correction factor for each q value.

    Notes
    -----
    Uses the angle-dependent absorption formula:
        nSi = (1 - exp(-tSi / (λ_Si * cos(theta)))) / (1 - exp(-tSi / λ_Si))
    where λ_Si is the X-ray attenuation length for silicon at the given energy.
    """
    Silen = Si_attenuation_length(keV)
    thetas = q2theta(q_arr, keV) # In this implementation, the theta here is the same as 2theta in Ma et al.
    nSi = (1 - np.exp(- tSi / (Silen * np.cos(thetas)))) / (1 - np.exp(- tSi / Silen))
    return nSi

def KaptonHN_correction(q_arr, keV, tK=8e-6):
    """
    Calculate the Kapton (HN) correction factor.

    Parameters
    ----------
    q_arr : array-like
        Array of momentum transfer (q) values.
    keV : float
        Photon energy in keV.
    tK : float, optional
        Kapton thickness in meters. Default is 8e-6 m.

    Returns
    -------
    numpy.ndarray
        Kapton correction factor for each q value.

    Notes
    -----
    Uses the angle-dependent exponential attenuation:
        nK = exp(-tK / (λ_K * cos(theta))) / exp(-tK / λ_K)
    where λ_K is the X-ray attenuation length for Kapton at the given energy.
    """
    Klen = KaptonHN_attenuation_length(keV)
    thetas = q2theta(q_arr, keV) # In this implementation, the theta here is the same as 2theta in Ma et al.
    nK = np.exp(- tK / (Klen * np.cos(thetas))) / np.exp(- tK / Klen)
    return nK

def Al_correction(q_arr, keV, tAl=4.5e-6):
    """
    Calculate the Aluminum correction factor.

    Parameters
    ----------
    q_arr : array-like
        Array of momentum transfer (q) values.
    keV : float
        Photon energy in keV.
    tAl : float, optional
        Aluminum thickness in meters. Default is 4.5e-6 m.

    Returns
    -------
    numpy.ndarray
        Aluminum correction factor for each q value.

    Notes
    -----
    Uses the angle-dependent exponential attenuation:
        nAl = exp(-tAl / (λ_Al * cos(theta))) / exp(-tAl / λ_Al)
    where λ_Al is the X-ray attenuation length for Aluminum at the given energy.
    """
    Allen = Al_attenuation_length(keV)
    thetas = q2theta(q_arr, keV) # In this implementation, the theta here is the same as 2theta in Ma et al.
    nAl = np.exp(- tAl / (Allen * np.cos(thetas))) / np.exp(- tAl / Allen)
    return nAl


def Be_correction(q_arr, keV, tBe=100e-6, rBe=250e-6, L=2.4e-3):
    """
    Calculate the Beryllium correction factor considering window geometry.

    Parameters
    ----------
    q_arr : array-like
        Array of momentum transfer (q) values.
    keV : float
        Photon energy in keV.
    tBe : float, optional
        Beryllium thickness in meters. Default is 100e-6 m.
    rBe : float, optional
        Radius of the Be window in meters. Default is 250e-6 m.
    L : float, optional
        Gas cell length in meters. Default is 2.4e-3 m.

    Returns
    -------
    numpy.ndarray
        Beryllium correction factor for each q value.

    Notes
    -----
    Accounts for partial path length through the Be window depending on
    scattering angle.
    """
    # Compute attenuation length and scattering angles
    Belen = Be_attenuation_length(keV)
    thetas = q2theta(q_arr, keV)  # same shape as q_arr

    # Compute partial path length through Be window
    xBe = np.minimum(rBe / np.tan(thetas), L)

    # Avoid division by zero for theta=0
    xBe = np.nan_to_num(xBe, nan=L, posinf=L, neginf=L)

    # Compute correction factor
    nBe = xBe / L + (L - xBe) / L * np.exp(-tBe / (Belen * np.cos(thetas)))
    return nBe

def cell_correction(q_arr, keV, tP=125e-6, rP=125e-6, L=2.4e-3):
    """
    Calculate the gas cell geometry correction factor.

    Parameters
    ----------
    q_arr : array-like
        Array of momentum transfer (q) values.
    keV : float
        Photon energy in keV.
    tP : float, optional
        Gas cell thickness in meters. Default is 125e-6 m.
    rP : float, optional
        Radius of the gas cell window in meters. Default is 125e-6 m.
    L : float, optional
        Gas cell length in meters. Default is 2.4e-3 m.

    Returns
    -------
    numpy.ndarray
        Cell geometry correction factor for each q value.

    Notes
    -----
    Accounts for the angle-dependent path length and geometry of the gas cell.
    """
    # Compute scattering angles
    thetas = q2theta(q_arr, keV)  # same shape as q_arr
    
    # Precompute helper array
    xmax = tP - rP / np.tan(thetas)
    
    # Mask to separate the two cases
    cond = np.tan(thetas) >= (rP / tP)
    
    # Initialize output
    nCell = np.empty_like(q_arr, dtype=float)
    
    # Case 1: tan(theta) >= rP/tP
    nCell[cond] = 1 + (rP / (L * np.tan(thetas[cond])))
    
    # Case 2: tan(theta) < rP/tP
    num = tP + (rP * xmax[~cond]) / (xmax[~cond] - rP)
    nCell[~cond] = 1 + num / L
    
    # Handle any division-by-zero or invalid values safely
    nCell = np.nan_to_num(nCell, nan=1.0, posinf=1.0, neginf=1.0)
    return nCell


def Si_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Silicon (Si) in meters.

    Parameters
    ----------
    keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Attenuation length of Silicon in meters.

    Notes
    -----
    Uses a spline interpolation of tabulated data. The input energy in keV
    is converted to eV for the interpolation, and the returned value is converted
    from microns to meters.
    """
    E_values, length = _load_attenuation_table('Si')
    Mu_Spline = InterpolatedUnivariateSpline(E_values, length)
    return Mu_Spline(keV*1000) * 1e-6  # Convert to m

def Al_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Aluminum (Al) for a given photon energy.

    Parameters
    ----------
    keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Attenuation length of Al in meters.

    Notes
    -----
    Uses a spline interpolation of tabulated data to compute the attenuation
    length. Tabulated values are converted from microns to meters.
    """
    E_values, length = _load_attenuation_table('Al')
    Mu_Spline = InterpolatedUnivariateSpline(E_values, length)
    return Mu_Spline(keV*1000) * 1e-6 # Convert to meters

def Be_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Beryllium (Be) for a given photon energy.

    Parameters
    ----------
    keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Attenuation length of Be in meters.

    Notes
    -----
    Computed via spline interpolation of tabulated attenuation data (in microns),
    which is converted to meters.
    """
    E_values, length = _load_attenuation_table('Be')
    Mu_Spline = InterpolatedUnivariateSpline(E_values, length)
    return Mu_Spline(keV*1000) * 1e-6  # Convert to m

def KaptonHN_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Kapton HN for a given photon energy.

    Parameters
    ----------
    keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Attenuation length of Kapton HN in meters.

    Notes
    -----
    Uses spline interpolation of tabulated attenuation lengths (in microns),
    then converts them to meters.
    """
    E_values, length = _load_attenuation_table('KaptonHN')
    Mu_Spline = InterpolatedUnivariateSpline(E_values, length)
    return Mu_Spline(keV*1000) * 1e-6  # Convert to m

def Zn_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Zinc (Zn) for a given photon energy.

    Parameters
    ----------
    keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Attenuation length of Zn in meters.

    Notes
    -----
    Uses spline interpolation of tabulated attenuation lengths (in microns),
    then converts them to meters.
    """
    with h5py.File(f"{_data_path}/Zn_attenuation_length.h5", 'r') as f:
        E_values = f['E_values'][:]
        length = f['length'][:]
    Mu_Spline = InterpolatedUnivariateSpline(E_values, length)
    return Mu_Spline(keV*1000) * 1e-6  # Convert to m

def J4M_efficiency(theta, keV, tSi = 318.5e-6, tAl = 4.5e-6, tK = 8e-6):
    """
    Calculate the detector efficiency of the Jungfrau4M detector for a given photon energy.

    Parameters
    ----------
    theta : float
        Scattering angle in radians.
    keV : float
        Photon energy in keV.
    tSi : float, optional
        Thickness of the silicon sensor in meters. Default is 318.5e-6.
    tAl : float, optional
        Thickness (total) of the aluminum layer + sputter coating in meters. Default is 4.5e-6.
    tK : float, optional
        Thickness of the kapton layer in meters. Default is 8e-6.

    Returns
    -------
    float
        Detector efficiency of the Jungfrau4M detector, from 0-1.

    Notes
    -----
    Uses spline interpolation of tabulated values to calculate the detector efficiency.
    """
    return (1-np.exp(-tSi /(np.cos(theta) * Si_attenuation_length(keV))))*(np.exp(-tK /(np.cos(theta) *KaptonHN_attenuation_length(keV)))) * (np.exp(-tAl /(np.cos(theta) * Al_attenuation_length(keV))))


def forward_scattering_correction(
    photon_energy_keV,
    formula,
    theory_q,
    theory_I_total,
    *,
    theta_max=1.0,
    n_theta=256,
    L=2.4e-3,
    t_Pt=125e-6,
    t_Pt_countersink=75e-6,
    r_Pt_entry=300e-6,
    r_Pt_bore=125e-6,
    t_Be=100e-6,
    r_Be=125e-6,
    t_K=8e-6,
    t_Al=4.5e-6,
    t_Si=318.5e-6,
    n_z=128,
    n_EF=80,
    EF_low_keV=None,
):
    """
    Build the angle-dependent scattering/detection correction C(theta) for a gas
    scattering cell with a beveled Pt entry pinhole, a Be exit pinhole, a
    Kapton/Al-sputter detector film, and a Si sensor.

    The model integrates over the scattering position z_s along the beam path --
    including gas that effuses into both pinhole bores (uniform density assumed) --
    and, for the Compton component, over the final-photon-energy distribution
    J(theta, E_F) obtained from `iam_compton_spectrum`. Energy dependence of the
    attenuation lengths of Be, Kapton, Al, and Si is resolved separately for
    elastic (at E=EI) and Compton (integrated over E_F) channels. The elastic
    and Compton fractions of the user's total theory pattern are derived from the
    molecular formula using the IAM inelastic intensity:

        f_co(q) = I_co_IAM(q) / I_total(q),   f_el(q) = 1 - f_co(q)

    (only the formula is needed; atomic coordinates are not, because Compton is
    incoherent in IAM.)

    The Pt pinhole is treated as a fully opaque absorber. Scattering from (0, 0, z_s)
    is accepted iff the forward-going ray clears the downstream Pt bore rim;
    for any angle below the bevel-limiting regime (~45 deg) the countersink never
    binds ahead of the straight bore, so the condition reduces to
        (t_Pt - z_s) * tan(theta) < r_Pt_bore   for z_s in [0, t_Pt].

    For Be, the ray traverses Be material between z_start = max(z_s, z_Be_start,
    z_s + r_Be/tan(theta)) and z_Be_end, with path length
    (z_Be_end - z_start)/cos(theta).

    The returned correction is normalized so that C(0) = 1.

    Parameters
    ----------
    photon_energy_keV : float
        Incident photon energy (EI) in keV.
    formula : str
        Chemical formula of the gas sample (e.g. "SF6"). Used to compute the
        IAM Compton intensity and the Compton double-differential profile.
    theory_q : array_like
        1D q grid of the user's total theory pattern, in inverse Angstroms.
    theory_I_total : array_like
        Total elastic + inelastic theory intensity on `theory_q`. May be computed
        at any desired level of theory; IAM is only used to estimate the
        Compton fraction.
    theta_max : float, optional
        Upper tabulation bound for the scattering angle in radians. Default 1.0.
    n_theta : int, optional
        Number of theta tabulation points. Default 256.
    L : float, optional
        Gas-cell length between pinhole inner faces, in meters. Default 2.4e-3.
    t_Pt : float, optional
        Total Pt pinhole thickness, in meters. Default 125e-6.
    t_Pt_countersink : float, optional
        Axial depth of the Pt countersink (upstream beveled cup), in meters.
        Default 75e-6. Retained for API clarity; not used at forward angles
        because the downstream straight bore always binds first.
    r_Pt_entry : float, optional
        Pt hole radius at the upstream (beam-entry) face, in meters. Default
        300e-6. Retained for API clarity; not used for the same reason as above.
    r_Pt_bore : float, optional
        Pt straight-bore radius (downstream 50 um of Pt), in meters. Default 125e-6.
    t_Be : float, optional
        Be pinhole thickness, in meters. Default 100e-6.
    r_Be : float, optional
        Be pinhole bore radius, in meters. Default 125e-6.
    t_K : float, optional
        Kapton film thickness, in meters. Default 8e-6.
    t_Al : float, optional
        Aluminum sputter-coating thickness, in meters. Default 4.5e-6.
    t_Si : float, optional
        Jungfrau4M Si-sensor thickness, in meters. Default 318.5e-6.
    n_z : int, optional
        Number of sampling points along z_s for the scattering-volume integration.
    n_EF : int, optional
        Number of sampling points for the Compton final-energy integration.
    EF_low_keV : float, optional
        Lower bound of the E_F grid, in keV. Default: EI - max(2, 0.3*EI).

    Returns
    -------
    C : InterpolatedUnivariateSpline
        Callable C(theta). Multiplying by the user's total theory pattern
        I_total(q(theta)) yields the predicted detector intensity up to the
        Thompson polarization factor, the per-pixel solid-angle factor, and an
        overall amplitude.

    Notes
    -----
    - Deposited-energy weighting (EF/EI) is included in the Compton channel,
      which is the appropriate convention for the Jungfrau4M operated in
      normal-gain (integrating) mode.
    - Thompson polarization and solid-angle effects are NOT included in C;
      apply them at the per-pixel level (see `model_full`).
    - At theta=0 the Compton kinematics collapse to a delta at E_F=EI and the
      profile integrated on a finite grid can be degenerate; the tabulation
      therefore starts at a tiny positive angle and the spline extrapolates
      smoothly down to theta=0.
    """
    from ..theory.iam import iam_compton_spectrum, iam_inelastic_pattern_from_formula

    EI = photon_energy_keV
    if EI <= 0:
        raise ValueError(f"photon_energy_keV must be positive, got {EI}.")

    # --- 1. Theta grid (start just above zero to keep Compton kinematics well-defined)
    theta_lo = 1e-6
    theta_grid = np.linspace(theta_lo, theta_max, n_theta)
    cos_th = np.cos(theta_grid)
    tan_th = np.tan(theta_grid)
    q_grid = theta2q(theta_grid, EI)

    # --- 2. Scattering-volume sampling
    L_total = t_Pt + L + t_Be
    z_Be_start = t_Pt + L
    z_Be_end = L_total
    z_s = (np.arange(n_z) + 0.5) * (L_total / n_z)

    theta_2d = theta_grid[:, None]      # (n_theta, 1)
    zs_2d = z_s[None, :]                # (1, n_z)
    tan_th_2d = np.tan(theta_2d)
    cos_th_2d = np.cos(theta_2d)

    # Pt clearance at downstream straight-bore rim
    pt_clear = np.where(
        zs_2d >= t_Pt,
        1.0,
        ((t_Pt - zs_2d) * tan_th_2d < r_Pt_bore).astype(float),
    )  # (n_theta, n_z)

    # Be path length in material (ray crosses inner wall at z_hit)
    tan_safe = np.where(tan_th_2d > 0, tan_th_2d, 1.0)
    z_hit = zs_2d + np.where(tan_th_2d > 0, r_Be / tan_safe, np.inf)
    z_start = np.maximum(np.maximum(zs_2d + 0.0 * tan_th_2d, z_Be_start), z_hit)
    Be_axial = np.maximum(0.0, z_Be_end - z_start)
    Be_path = Be_axial / cos_th_2d      # (n_theta, n_z)

    # --- 3. Elastic detection efficiency A_el(theta)
    lam_Be_EI = Be_attenuation_length(EI)
    lam_K_EI = KaptonHN_attenuation_length(EI)
    lam_Al_EI = Al_attenuation_length(EI)
    lam_Si_EI = Si_attenuation_length(EI)

    T_Be_EI = np.exp(-Be_path / lam_Be_EI)
    T_K_EI = np.exp(-t_K / (lam_K_EI * cos_th))
    T_Al_EI = np.exp(-t_Al / (lam_Al_EI * cos_th))
    eta_Si_EI = 1.0 - np.exp(-t_Si / (lam_Si_EI * cos_th))

    A_el = (pt_clear * T_Be_EI).mean(axis=1) * T_K_EI * T_Al_EI * eta_Si_EI

    # --- 4. Compton-averaged detection efficiency A_co(theta)
    if EF_low_keV is None:
        EF_low_keV = max(EI - max(2.0, 0.3 * EI), 0.5)
    EF_grid = np.linspace(EF_low_keV, EI, n_EF)

    J = iam_compton_spectrum(formula, theta_grid, EI, EF_grid)
    if J.ndim == 1:
        J = J[None, :]
    J_sum = np.trapz(J, EF_grid, axis=1)
    degenerate = J_sum <= 0
    J_norm = np.where(
        J_sum[:, None] > 0,
        J / np.where(J_sum[:, None] > 0, J_sum[:, None], 1.0),
        0.0,
    )  # (n_theta, n_EF)

    lam_Be_EF = np.array([Be_attenuation_length(e) for e in EF_grid])
    lam_K_EF = np.array([KaptonHN_attenuation_length(e) for e in EF_grid])
    lam_Al_EF = np.array([Al_attenuation_length(e) for e in EF_grid])
    lam_Si_EF = np.array([Si_attenuation_length(e) for e in EF_grid])

    Be_path_3d = Be_path[:, :, None]           # (n_theta, n_z, 1)
    pt_clear_3d = pt_clear[:, :, None]
    T_Be_EF_3d = np.exp(-Be_path_3d / lam_Be_EF[None, None, :])
    T_Be_avg = (pt_clear_3d * T_Be_EF_3d).mean(axis=1)   # (n_theta, n_EF)

    cos_th_col = cos_th[:, None]
    T_K_EF = np.exp(-t_K / (lam_K_EF[None, :] * cos_th_col))
    T_Al_EF = np.exp(-t_Al / (lam_Al_EF[None, :] * cos_th_col))
    eta_Si_EF = 1.0 - np.exp(-t_Si / (lam_Si_EF[None, :] * cos_th_col))
    E_weight = EF_grid[None, :] / EI

    integrand_co = J_norm * E_weight * T_Be_avg * T_K_EF * T_Al_EF * eta_Si_EF
    A_co = np.trapz(integrand_co, EF_grid, axis=1)

    # At theta where J_sum collapsed (near theta=0), Compton is kinematically
    # indistinguishable from elastic; fall back to A_el.
    A_co = np.where(degenerate, A_el, A_co)

    # --- 5. IAM-derived elastic/Compton split of the user's total pattern
    I_co_IAM = np.maximum(iam_inelastic_pattern_from_formula(formula, q_grid), 0.0)
    I_total_spline = InterpolatedUnivariateSpline(
        np.asarray(theory_q, dtype=float),
        np.asarray(theory_I_total, dtype=float),
        ext=3,
    )
    I_total_on_grid = I_total_spline(q_grid)
    # Guard against tiny/zero denominators
    denom = np.where(I_total_on_grid > 1e-30, I_total_on_grid, 1e-30)
    f_co = np.clip(I_co_IAM / denom, 0.0, 1.0)
    f_el = 1.0 - f_co

    # --- 6. Assemble and normalize so C(0) = 1 (extrapolating below theta_lo)
    numerator = f_el * A_el + f_co * A_co
    C_values = numerator / numerator[0]

    return InterpolatedUnivariateSpline(theta_grid, C_values, ext=3)