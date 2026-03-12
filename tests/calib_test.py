# %%
import xrayscatteringtools.calib.scattering_corrections
import matplotlib.pyplot as plt
import numpy as np

keV = 15.155

q = np.linspace(0.1, 4, 1000)

cell_factor = xrayscatteringtools.calib.scattering_corrections.cell_correction(q, keV)
Be_factor = xrayscatteringtools.calib.scattering_corrections.Be_correction(q, keV)
Al_factor = xrayscatteringtools.calib.scattering_corrections.Al_correction(q, keV)
Kapton_factor = xrayscatteringtools.calib.scattering_corrections.KaptonHN_correction(q, keV)
Si_factor = xrayscatteringtools.calib.scattering_corrections.Si_correction(q, keV)
total_factor = xrayscatteringtools.calib.scattering_corrections.correction_factor(q, keV)

plt.figure(figsize=(18,10))
plt.subplot(2,3,1)
plt.plot(q, cell_factor, label='Cell Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Cell Correction Factor')
plt.subplot(2,3,2)
plt.plot(q, Be_factor, label='Be Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Be Correction Factor')
plt.subplot(2,3,3)
plt.plot(q, Al_factor, label='Al Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Al Correction Factor')
plt.subplot(2,3,4)
plt.plot(q, Kapton_factor, label='Kapton Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Kapton Correction Factor')
plt.subplot(2,3,5)
plt.plot(q, Si_factor, label='Si Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Si Correction Factor')
plt.subplot(2,3,6)
plt.plot(q, total_factor, label='Total Correction Factor')
plt.xlabel('q (1/Angstrom)')
plt.ylabel('Correction Factor')
plt.title('Total Correction Factor')

# %%
# Cross reference these with henke and Ma et al. (2024)
print(f"Be: {xrayscatteringtools.calib.scattering_corrections.Be_attenuation_length(15.155)*1e6}")
print(f"Al: {xrayscatteringtools.calib.scattering_corrections.Al_attenuation_length(15.155)*1e6}")
print(f"Si: {xrayscatteringtools.calib.scattering_corrections.Si_attenuation_length(15.155)*1e6}")
print(f"KaptonHN: {xrayscatteringtools.calib.scattering_corrections.KaptonHN_attenuation_length(15.155)*1e6}")

# %%
plt.figure(figsize=(6,4))
keV = 10
q = np.linspace(0.1, 4, 1000)
total_factor = xrayscatteringtools.calib.scattering_corrections.correction_factor(q, keV)
plt.plot(q, total_factor, label='10 keV')
keV = 15
q = np.linspace(0.1, 6.7, 1000)
total_factor = xrayscatteringtools.calib.scattering_corrections.correction_factor(q, keV)
plt.plot(q, total_factor, label='15 keV')
keV = 18
q = np.linspace(0.1, 8.9, 1000)
total_factor = xrayscatteringtools.calib.scattering_corrections.correction_factor(q, keV)
plt.plot(q, total_factor, label='18 keV')
keV = 20
q = np.linspace(0.1, 10, 1000)
total_factor = xrayscatteringtools.calib.scattering_corrections.correction_factor(q, keV)
plt.plot(q, total_factor, label='20 keV')
plt.legend()
plt.title('Recreated figure from Ma et al (2024).')
plt.xlabel(r'q ($\mathrm{Å^{-1}}$)')
plt.ylabel(r'$\eta_{AoS}(2\theta)$')
plt.show()

# %%
theta = np.linspace(0, np.pi/4, 100)
keV = 15
plt.plot(theta, xrayscatteringtools.calib.scattering_corrections.J4M_efficiency(theta,keV))
plt.xlabel('Scattering Angle (rad)')
plt.title(f'J4M Efficiency at {keV} keV')
plt.ylabel('J4M Efficiency')
plt.show()

# %%
e_range = np.linspace(6, 10, 100)
plt.plot(e_range, xrayscatteringtools.calib.scattering_corrections.Zn_attenuation_length(e_range)*1e6)
plt.xlabel('Energy (keV)')
plt.title('Zn Attenuation Length')
plt.ylabel('Attenuation Length (microns)')
plt.show()
