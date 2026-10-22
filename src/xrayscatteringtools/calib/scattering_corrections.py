from ..utils import q2theta
import numpy as np
import xraylib


def correction_factor(
    q_arr,
    keV,
    L=2.4e-3,
    tSi=318.5e-6,
    tK=8e-6,
    tAl=4.5e-6,
    tBe=100e-6,
    rBe=125e-6,
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
    q_arr : array-like
        Array of momentum transfer (q) values in Å⁻¹.
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
        Hole radius of the Be exit pinhole in meters. Default is 125e-6 m (250 µm diameter).
    tP : float, optional
        Thickness of the gas cell platinum pinhole in meters. Default is 125e-6 m.
    rP : float, optional
        Hole radius of the gas cell platinum pinhole in meters. Default is 125e-6 m (250 µm diameter).

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

    See Lingyu Ma et al 2024 J. Phys. B: At. Mol. Opt. Phys. 57 205602
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


def Be_correction(q_arr, keV, tBe=100e-6, rBe=125e-6, L=2.4e-3):
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
        Hole radius of the Be exit pinhole in meters. Default is 125e-6 m (250 µm diameter).
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
    with np.errstate(divide='ignore'):
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
        Thickness of the gas cell platinum pinhole in meters. Default is 125e-6 m.
    rP : float, optional
        Hole radius of the gas cell platinum pinhole in meters. Default is 125e-6 m (250 µm diameter).
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
    q_arr = np.asarray(q_arr, dtype=float)
    # Compute scattering angles
    thetas = q2theta(q_arr, keV)  # same shape as q_arr

    with np.errstate(divide='ignore', invalid='ignore'):
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


# Attenuation lengths come from xraylib. The mass attenuation coefficient (cm^2/g) times the
# density (g/cm^3) gives the linear attenuation coefficient (1/cm); its inverse, times 1e-2,
# is the attenuation length in meters. The windows use the total cross section, since
# scattering also removes photons from the path. The Si sensor uses photoabsorption only,
# since only absorbed photons deposit charge. xraylib takes one energy at a time, so the
# functions are wrapped in np.vectorize to also accept arrays.

def Si_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Silicon (Si) in meters.

    Parameters
    ----------
    keV : float or array-like
        Photon energy in keV.

    Returns
    -------
    float or numpy.ndarray
        Attenuation length of Silicon in meters.

    Notes
    -----
    Photoabsorption only (xraylib ``CS_Photo``), density 2.33 g/cm^3.
    """
    keV = np.asarray(keV, dtype=float)
    return 1e-2 / (np.vectorize(xraylib.CS_Photo)(14, keV) * 2.33)

def Al_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Aluminum (Al) in meters.

    Parameters
    ----------
    keV : float or array-like
        Photon energy in keV.

    Returns
    -------
    float or numpy.ndarray
        Attenuation length of Al in meters.

    Notes
    -----
    Total cross section (xraylib ``CS_Total``), density 2.699 g/cm^3.
    """
    keV = np.asarray(keV, dtype=float)
    return 1e-2 / (np.vectorize(xraylib.CS_Total)(13, keV) * 2.699)

def Be_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Beryllium (Be) in meters.

    Parameters
    ----------
    keV : float or array-like
        Photon energy in keV.

    Returns
    -------
    float or numpy.ndarray
        Attenuation length of Be in meters.

    Notes
    -----
    Total cross section (xraylib ``CS_Total``), density 1.848 g/cm^3.
    """
    keV = np.asarray(keV, dtype=float)
    return 1e-2 / (np.vectorize(xraylib.CS_Total)(4, keV) * 1.848)

def KaptonHN_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Kapton HN in meters.

    Parameters
    ----------
    keV : float or array-like
        Photon energy in keV.

    Returns
    -------
    float or numpy.ndarray
        Attenuation length of Kapton HN in meters.

    Notes
    -----
    Total cross section (xraylib ``CS_Total_CP``) of C22H10N2O5, density 1.42 g/cm^3.
    """
    keV = np.asarray(keV, dtype=float)
    return 1e-2 / (np.vectorize(xraylib.CS_Total_CP)("C22H10N2O5", keV) * 1.42)

def Zn_attenuation_length(keV):
    """
    Calculate the X-ray attenuation length of Zinc (Zn) in meters.

    Parameters
    ----------
    keV : float or array-like
        Photon energy in keV.

    Returns
    -------
    float or numpy.ndarray
        Attenuation length of Zn in meters.

    Notes
    -----
    Total cross section (xraylib ``CS_Total``), density 7.14 g/cm^3.
    """
    keV = np.asarray(keV, dtype=float)
    return 1e-2 / (np.vectorize(xraylib.CS_Total)(30, keV) * 7.14)

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
    The fraction absorbed in the Si sensor times the transmission through the Kapton and Al layers.
    """
    return (1-np.exp(-tSi /(np.cos(theta) * Si_attenuation_length(keV))))*(np.exp(-tK /(np.cos(theta) *KaptonHN_attenuation_length(keV)))) * (np.exp(-tAl /(np.cos(theta) * Al_attenuation_length(keV))))
