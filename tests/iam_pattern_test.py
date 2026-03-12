# %%
from xrayscatteringtools.theory.iam import iam_total_pattern, iam_inelastic_pattern, iam_elastic_pattern, iam_compton_spectrum
import matplotlib.pyplot as plt
import numpy as np

q_arr = np.linspace(0, 10, 1000)  # Example q values
total_pattern = iam_total_pattern('DMP_ground.xyz', q_arr)
elastic_pattern = iam_elastic_pattern('DMP_ground.xyz', q_arr)
inelastic_pattern = iam_inelastic_pattern('DMP_ground.xyz', q_arr)

plt.figure(figsize=(10, 6))
plt.plot(q_arr, total_pattern, label='Total Pattern')
plt.plot(q_arr, inelastic_pattern, label='Inelastic Pattern')
plt.plot(q_arr, elastic_pattern, label='Elastic Pattern')
plt.xlabel('Q (1/Å)')
plt.ylabel('Intensity (a.u.)')
plt.title('X-ray Scattering Patterns')
plt.legend()
plt.grid()
plt.show()

# %%
from xrayscatteringtools.theory.iam import iam_total_pattern, iam_inelastic_pattern, iam_elastic_pattern
import matplotlib.pyplot as plt
import numpy as np

q_arr = np.linspace(0, 10, 1000)  # Example q values
DMP_ground = iam_total_pattern('DMP_ground.xyz', q_arr)
DMP_L1 = iam_total_pattern('DMP_L1.xyz', q_arr)
DMP_L2 = iam_total_pattern('DMP_L2.xyz', q_arr)
DMP_D = iam_total_pattern('DMP_D.xyz', q_arr)

plt.figure(figsize=(10, 6))
plt.plot(q_arr, DMP_ground*q_arr, label='DMP Ground')
plt.plot(q_arr, DMP_L1*q_arr, label='DMP L1')
plt.plot(q_arr, DMP_L2*q_arr, label='DMP L2')
plt.plot(q_arr, DMP_D*q_arr, label='DMP D')
plt.xlabel('q (1/Å)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.grid()

# %%
pdiff_DMP_L1 = (DMP_L1 - DMP_ground) / DMP_ground
pdiff_DMP_L2 = (DMP_L2 - DMP_ground) / DMP_ground
pdiff_DMP_D = (DMP_D - DMP_ground) / DMP_ground

plt.figure(figsize=(10, 6))
plt.plot(q_arr, pdiff_DMP_L1, label='DMP L1')
plt.plot(q_arr, pdiff_DMP_L2, label='DMP L2')
plt.plot(q_arr, pdiff_DMP_D, label='DMP D')
plt.xlabel('q (1/Å)')
plt.ylabel('% Difference')
plt.legend()
plt.grid()
plt.xlim(0,4)
plt.show()

# %%
from xrayscatteringtools.theory.iam import iam_compton_spectrum
import matplotlib.pyplot as plt
import numpy as np

energy_values = np.linspace(8, 9.666, 1000)
compton_spectrum = iam_compton_spectrum('SF6', 30, 9.666, energy_values)
plt.plot(energy_values, compton_spectrum)
plt.show()

# %%
from xrayscatteringtools.theory.iam import iam_compton_spectrum
import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, np.pi, 1000)
energy_values = np.linspace(8.8, 9.666, 1000)
compton_spectrum = iam_compton_spectrum('SF6', theta, 9.666, energy_values)
plt.pcolormesh(energy_values, theta*180/np.pi, compton_spectrum, shading='auto')
plt.xlabel('Scattered Photon Energy (keV)', fontsize=12)
plt.ylabel(r'Scattering angle $\theta$', fontsize=12)
plt.show()
