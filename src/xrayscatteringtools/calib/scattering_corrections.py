from scipy.interpolate import InterpolatedUnivariateSpline
from xrayscatteringtools.utils import q2theta
import numpy as np

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
    E_values = [
        6000.00, 6140.00, 6280.00, 6420.00, 6560.00, 6700.00, 6840.00, 6980.00,
        7120.00, 7260.00, 7400.00, 7540.00, 7680.00, 7820.00, 7960.00, 8100.00,
        8240.00, 8380.00, 8520.00, 8660.00, 8800.00, 8940.00, 9080.00, 9220.00,
        9360.00, 9500.00, 9640.00, 9780.00, 9920.00, 10060.0, 10200.0, 10340.0,
        10480.0, 10620.0, 10760.0, 10900.0, 11040.0, 11180.0, 11320.0, 11460.0,
        11600.0, 11740.0, 11880.0, 12020.0, 12160.0, 12300.0, 12440.0, 12580.0,
        12720.0, 12860.0, 13000.0, 13140.0, 13280.0, 13420.0, 13560.0, 13700.0,
        13840.0, 13980.0, 14120.0, 14260.0, 14400.0, 14540.0, 14680.0, 14820.0,
        14960.0, 15100.0, 15240.0, 15380.0, 15520.0, 15660.0, 15800.0, 15940.0,
        16080.0, 16220.0, 16360.0, 16500.0, 16640.0, 16780.0, 16920.0, 17060.0,
        17200.0, 17340.0, 17480.0, 17620.0, 17760.0, 17900.0, 18040.0, 18180.0,
        18320.0, 18460.0, 18600.0, 18740.0, 18880.0, 19020.0, 19160.0, 19300.0,
        19440.0, 19580.0, 19720.0, 19860.0, 20000.0
    ] # eV

    length = [
        30.2985, 32.3736, 34.5409, 36.8058, 39.1665, 41.6272, 44.1903, 46.8565,
        49.6295, 52.5100, 55.5013, 58.6048, 61.8219, 65.1565, 68.6095, 72.1834,
        75.8799, 79.7013, 83.6497, 87.7275, 91.9370, 96.2790, 100.757, 105.373,
        110.128, 115.025, 120.067, 125.254, 130.590, 136.076, 141.715, 147.509,
        153.460, 159.570, 165.839, 172.274, 178.872, 185.637, 192.574, 199.679,
        206.960, 214.417, 222.051, 229.865, 237.861, 246.040, 254.404, 262.959,
        271.704, 280.639, 289.770, 299.096, 308.619, 318.347, 328.274, 338.404,
        348.743, 359.292, 370.049, 381.017, 392.198, 403.601, 415.221, 427.057,
        439.120, 451.410, 463.930, 476.677, 489.653, 502.866, 516.312, 529.997,
        543.916, 558.080, 572.484, 587.133, 602.027, 617.169, 632.558, 648.196,
        664.092, 680.241, 696.650, 713.315, 730.242, 747.427, 764.877, 782.593,
        800.574, 818.826, 837.346, 856.145, 875.212, 894.559, 914.172, 934.073,
        954.240, 974.706, 995.446, 1016.48, 1037.80
    ] # Micron
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
    E_values = [
        6000.00, 6140.00, 6280.00, 6420.00, 6560.00, 6700.00, 6840.00, 6980.00,
        7120.00, 7260.00, 7400.00, 7540.00, 7680.00, 7820.00, 7960.00, 8100.00,
        8240.00, 8380.00, 8520.00, 8660.00, 8800.00, 8940.00, 9080.00, 9220.00,
        9360.00, 9500.00, 9640.00, 9780.00, 9920.00, 10060.0, 10200.0, 10340.0,
        10480.0, 10620.0, 10760.0, 10900.0, 11040.0, 11180.0, 11320.0, 11460.0,
        11600.0, 11740.0, 11880.0, 12020.0, 12160.0, 12300.0, 12440.0, 12580.0,
        12720.0, 12860.0, 13000.0, 13140.0, 13280.0, 13420.0, 13560.0, 13700.0,
        13840.0, 13980.0, 14120.0, 14260.0, 14400.0, 14540.0, 14680.0, 14820.0,
        14960.0, 15100.0, 15240.0, 15380.0, 15520.0, 15660.0, 15800.0, 15940.0,
        16080.0, 16220.0, 16360.0, 16500.0, 16640.0, 16780.0, 16920.0, 17060.0,
        17200.0, 17340.0, 17480.0, 17620.0, 17760.0, 17900.0, 18040.0, 18180.0,
        18320.0, 18460.0, 18600.0, 18740.0, 18880.0, 19020.0, 19160.0, 19300.0,
        19440.0, 19580.0, 19720.0, 19860.0, 20000.0
    ] # eV

    length = [
        33.5963, 35.9138, 38.3384, 40.8699, 43.5113, 46.2649, 49.1331, 52.1183,
        55.2230, 58.4497, 61.8005, 65.2780, 68.8848, 72.6228, 76.4952, 80.5027,
        84.6497, 88.9378, 93.3698, 97.9478, 102.673, 107.549, 112.580, 117.765,
        123.108, 128.611, 134.276, 140.108, 146.102, 152.272, 158.613, 165.131,
        171.824, 178.696, 185.752, 192.991, 200.418, 208.034, 215.842, 223.844,
        232.042, 240.437, 249.034, 257.835, 266.841, 276.056, 285.482, 295.117,
        304.970, 315.038, 325.324, 335.837, 346.572, 357.532, 368.723, 380.144,
        391.796, 403.686, 415.811, 428.178, 440.785, 453.638, 466.735, 480.075,
        493.674, 507.531, 521.648, 536.026, 550.660, 565.562, 580.723, 596.154,
        611.848, 627.816, 644.054, 660.571, 677.362, 694.432, 711.784, 729.418,
        747.339, 765.542, 784.034, 802.811, 821.888, 841.256, 860.921, 880.880,
        901.138, 921.698, 942.559, 963.730, 985.201, 1006.98, 1029.06, 1051.47,
        1074.17, 1097.20, 1120.53, 1144.28, 1168.39
    ] # Micron
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
    E_values = [
        6000.00, 6140.00, 6280.00, 6420.00, 6560.00, 6700.00, 6840.00, 6980.00,
        7120.00, 7260.00, 7400.00, 7540.00, 7680.00, 7820.00, 7960.00, 8100.00,
        8240.00, 8380.00, 8520.00, 8660.00, 8800.00, 8940.00, 9080.00, 9220.00,
        9360.00, 9500.00, 9640.00, 9780.00, 9920.00, 10060.0, 10200.0, 10340.0,
        10480.0, 10620.0, 10760.0, 10900.0, 11040.0, 11180.0, 11320.0, 11460.0,
        11600.0, 11740.0, 11880.0, 12020.0, 12160.0, 12300.0, 12440.0, 12580.0,
        12720.0, 12860.0, 13000.0, 13140.0, 13280.0, 13420.0, 13560.0, 13700.0,
        13840.0, 13980.0, 14120.0, 14260.0, 14400.0, 14540.0, 14680.0, 14820.0,
        14960.0, 15100.0, 15240.0, 15380.0, 15520.0, 15660.0, 15800.0, 15940.0,
        16080.0, 16220.0, 16360.0, 16500.0, 16640.0, 16780.0, 16920.0, 17060.0,
        17200.0, 17340.0, 17480.0, 17620.0, 17760.0, 17900.0, 18040.0, 18180.0,
        18320.0, 18460.0, 18600.0, 18740.0, 18880.0, 19020.0, 19160.0, 19300.0,
        19440.0, 19580.0, 19720.0, 19860.0, 20000.0
    ]

    length = [
        2236.49, 2400.58, 2571.82, 2751.11, 2938.12, 3132.45, 3334.08, 3542.98,
        3759.08, 3982.32, 4212.59, 4449.79, 4693.79, 4944.48, 5201.64, 5465.44,
        5735.56, 6012.75, 6296.96, 6586.83, 6882.13, 7182.60, 7487.94, 7797.85,
        8112.04, 8430.21, 8752.01, 9077.13, 9405.23, 9737.18, 10073.2, 10411.6,
        10752.0, 11094.1, 11437.5, 11782.0, 12127.2, 12472.7, 12818.3, 13163.7,
        13508.5, 13852.4, 14195.1, 14536.5, 14876.0, 15213.7, 15549.8, 15885.7,
        16218.9, 16549.2, 16876.3, 17200.0, 17520.2, 17836.7, 18149.3, 18457.8,
        18762.0, 19062.0, 19357.5, 19648.4, 19934.7, 20216.2, 20492.9, 20764.7,
        21031.4, 21302.6, 21572.9, 21838.7, 22099.9, 22356.7, 22608.8, 22856.4,
        23099.4, 23337.8, 23572.2, 23803.1, 24029.4, 24251.3, 24468.5, 24681.1,
        24889.3, 25093.0, 25292.3, 25487.3, 25678.0, 25864.3, 26046.5, 26224.5,
        26398.4, 26568.2, 26734.0, 26895.9, 27054.0, 27208.2, 27358.8, 27505.5,
        27648.7, 27788.3, 27924.5, 28057.2, 28186.6
    ]

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
    E_values = [
        6000.00, 6140.00, 6280.00, 6420.00, 6560.00, 6700.00, 6840.00, 6980.00,
        7120.00, 7260.00, 7400.00, 7540.00, 7680.00, 7820.00, 7960.00, 8100.00,
        8240.00, 8380.00, 8520.00, 8660.00, 8800.00, 8940.00, 9080.00, 9220.00,
        9360.00, 9500.00, 9640.00, 9780.00, 9920.00, 10060.0, 10200.0, 10340.0,
        10480.0, 10620.0, 10760.0, 10900.0, 11040.0, 11180.0, 11320.0, 11460.0,
        11600.0, 11740.0, 11880.0, 12020.0, 12160.0, 12300.0, 12440.0, 12580.0,
        12720.0, 12860.0, 13000.0, 13140.0, 13280.0, 13420.0, 13560.0, 13700.0,
        13840.0, 13980.0, 14120.0, 14260.0, 14400.0, 14540.0, 14680.0, 14820.0,
        14960.0, 15100.0, 15240.0, 15380.0, 15520.0, 15660.0, 15800.0, 15940.0,
        16080.0, 16220.0, 16360.0, 16500.0, 16640.0, 16780.0, 16920.0, 17060.0,
        17200.0, 17340.0, 17480.0, 17620.0, 17760.0, 17900.0, 18040.0, 18180.0,
        18320.0, 18460.0, 18600.0, 18740.0, 18880.0, 19020.0, 19160.0, 19300.0,
        19440.0, 19580.0, 19720.0, 19860.0, 20000.0
    ]

    length = [
        482.639, 518.357, 555.868, 595.186, 636.377, 679.478, 724.515, 771.554,
        820.639, 871.805, 925.074, 980.527, 1038.17, 1098.07, 1160.24, 1224.76,
        1291.66, 1360.97, 1432.71, 1506.92, 1583.63, 1662.89, 1744.71, 1829.12,
        1916.14, 2005.81, 2098.15, 2193.20, 2290.93, 2391.48, 2494.84, 2601.00,
        2709.95, 2821.71, 2936.29, 3053.69, 3173.94, 3297.03, 3422.98, 3551.74,
        3683.35, 3817.78, 3955.02, 4095.08, 4237.92, 4383.57, 4531.99, 4683.10,
        4836.97, 4993.50, 5152.65, 5314.53, 5478.98, 5645.95, 5815.49, 5987.49,
        6161.89, 6338.73, 6517.89, 6699.35, 6883.04, 7068.94, 7256.91, 7446.90,
        7638.99, 7833.72, 8030.71, 8229.62, 8430.37, 8632.92, 8837.16, 9043.03,
        9250.44, 9459.37, 9669.72, 9881.46, 10094.5, 10308.7, 10524.0, 10740.4,
        10957.8, 11176.1, 11395.2, 11615.0, 11835.5, 12056.6, 12278.2, 12500.2,
        12722.5, 12945.2, 13168.0, 13391.0, 13614.1, 13837.0, 14060.0, 14282.6,
        14505.1, 14727.2, 14948.9, 15160.9, 15367.5
]

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