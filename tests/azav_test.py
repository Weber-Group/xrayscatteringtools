# %%
# Testing the azav code in utils.py
import xrayscatteringtools as xrst
from xrayscatteringtools.calib import model
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# %%
CCSD = xrst.theory.SF6__CCSD__aug_cc_pVDZ
theory = InterpolatedUnivariateSpline(CCSD.q, CCSD.I_q)
proj = model(
    [xrst.J4M.x, xrst.J4M.y],
    1,
    0,
    0,
    90_000,
    0,
    9.666,
    theory,
    do_angle_of_scattering_correction=False,
)

# %%
xrst.plot_j4m(proj)

# %%
_kwargs = {
    'x': xrst.J4M.x,
    'y': xrst.J4M.y,
    'x0':  0, # -23.70994492481425
    'y0': 0, # -42.29215571926531
    'z0': 90_000, # 95104.38014082455
    'mask': np.zeros_like(xrst.J4M.x).astype(bool),  # & ~ndi.binary_closing((diff <-20) | (diff > 20),structure=np.ones((1, 3, 3), dtype=bool)),
    'keV': 9.666,
    'pPlane':  0,
    'qBin': 0.005,
}
q, azav = xrst.azimuthalBinning(proj, **_kwargs)
plt.plot(q,azav, label='Azav from proj')
plt.plot(q,theory(q),label='Pure Theory')
plt.legend()

# %%
plt.plot(q,q*(theory(q)-azav)/theory(q))

# %%
_kwargs_fai = {
    'x0':  0, # -23.70994492481425
    'y0': 0, # -42.29215571926531
    'z0': 90_000, # 95104.38014082455
    'mask': np.zeros_like(xrst.J4M.x).astype(bool),  # & ~ndi.binary_closing((diff <-20) | (diff > 20),structure=np.ones((1, 3, 3), dtype=bool)),
    'keV': 9.666,
    'pPlane':  0,
    'qBin': 0.005,
}
q_fai, azav_fai = xrst.azimuthalBinning_pyfai(proj, **_kwargs_fai)
plt.plot(q_fai, azav_fai, label='Azav from proj')
plt.plot(q,theory(q),label='Pure Theory')
plt.legend()

# %%
plt.plot(q_fai,(theory(q_fai)-azav_fai)/theory(q_fai), label='Azav from pyFAI')
plt.plot(q,(theory(q)-azav)/theory(q),label='Azav from custom code')
plt.legend()
plt.show()

# %% Testing speeds
_kwargs_fai = {
    'x0':  0, # -23.70994492481425
    'y0': 0, # -42.29215571926531
    'z0': 90_000, # 95104.38014082455
    'mask': np.zeros_like(xrst.J4M.x).astype(bool),  # & ~ndi.binary_closing((diff <-20) | (diff > 20),structure=np.ones((1, 3, 3), dtype=bool)),
    'keV': 9.666,
    'pPlane':  0,
    'qBin': 0.005,
}

_kwargs = {
    'x': xrst.J4M.x,
    'y': xrst.J4M.y,
    'x0':  0, # -23.70994492481425
    'y0': 0, # -42.29215571926531
    'z0': 90_000, # 95104.38014082455
    'mask': np.zeros_like(xrst.J4M.x).astype(bool),  # & ~ndi.binary_closing((diff <-20) | (diff > 20),structure=np.ones((1, 3, 3), dtype=bool)),
    'keV': 9.666,
    'pPlane':  0,
    'qBin': 0.005,
}

# %% Timing the custom azimuthalBinning method
num_iterations = 10
from time import time
start = time()
for _ in range(num_iterations):
    q, azav = xrst.azimuthalBinning(proj, **_kwargs)
end = time()
custom_total_time = end - start
print(f"Custom code took {custom_total_time:.2f} seconds total, or {custom_total_time / num_iterations:.2f} seconds per iteration")

# Timing the pyFAI azimuthalBinning_pyfai method
start = time()
for _ in range(num_iterations):
    q_fai, azav_fai = xrst.azimuthalBinning_pyfai(proj, **_kwargs_fai)
end = time()
pyfai_total_time = end - start
print(f"pyFAI code took {pyfai_total_time:.2f} seconds total, or {pyfai_total_time / num_iterations:.2f} seconds per iteration")
# %%
