# %%
import numpy as np
import matplotlib.pyplot as plt
from xrayscatteringtools.theory.iam import iam_elastic_pattern_oriented, iam_inelastic_pattern_oriented, iam_total_pattern_oriented, iam_elastic_pattern, iam_inelastic_pattern, iam_total_pattern
from xrayscatteringtools.utils import rotate_molecule
from xrayscatteringtools.io import write_xyz, read_xyz
from xrayscatteringtools.theory.geometries import SF6__CCSD_T_DHK__aug_cc_pV5Z_DK as sf6

# %%
########## Parameters ##########
N = 10 # Number of Molecules
################################
# Read the xyz file and then rotate the molecule by euler angles
atoms = sf6.atoms
coords = sf6.geometry
np.random.seed(42)  # For reproducibility

# all_atoms = []
# all_coords = []
pattern = np.zeros((100, 180))  # Initialize pattern array
q_arr = np.linspace(0, 8, 100)      # Radial coordinate (momentum transfer q)
phi_arr = np.linspace(0, 2 * np.pi, 180) # Angular coordinate (azimuthal angle φ)

for i in range(N):
    # Random Euler angles
    rand_alpha = np.random.uniform(0, 360)
    rand_beta = np.random.uniform(0, 360)
    rand_gamma = np.random.uniform(0, 360)
    # Rotate molecule
    rand_rotated_coords = rotate_molecule(coords, rand_alpha, rand_beta, rand_gamma)
    # # Random translation vector, each component in [-10, 10], but ensure norm > 3
    # while True:
    #     translation_vector = np.random.uniform(-10, 10, size=3)
    #     if np.linalg.norm(translation_vector) > 3:
    #         break
    # translated_coords = translate_molecule(rand_rotated_coords, translation_vector)
    # # Append atoms and coordinates for this molecule
    # all_atoms.extend(atoms)
    # all_coords.extend(translated_coords)

    # Define the coordinates for the grid
    write_xyz('sf6_test.xyz', f'{N} random orientations and translations', atoms, np.array(rand_rotated_coords))
    pattern += iam_total_pattern_oriented('sf6_test.xyz', q_arr, phi_arr)
pattern /= N  # Average the pattern to the number of molecules
# Create a figure and an axes object with a polar projection
plt.figure(figsize=(15, 5))
plt.yticks([])
plt.xticks([])
plt.box(False)
plt.title('Total Scattering Pattern (q·I(q,φ))')
plt.subplot(1,2,1,projection='polar')
# Use pcolormesh to create the plot.
plt.pcolormesh(phi_arr, q_arr, pattern * q_arr[:, np.newaxis], shading='gouraud', cmap='viridis')
plt.grid(False)
plt.xticks([])  # Hide angular tick labels
plt.yticks([])  # Hide radial tick labels
# plt.title('Oriented Elastic Scattering Pattern (q·I(q,φ))')

plt.subplot(1,2,2)
plt.plot(q_arr, pattern.mean(axis=1)*q_arr,'-',label='Oriented Average')
plt.plot(q_arr, iam_total_pattern(sf6,q_arr)*q_arr,'--',label='IAM')
plt.legend()
plt.xlabel(r'q ($Å^{-1}$)')
plt.ylabel(r'q·I$_{Tot}$(q) (arb. units)')
plt.show()

# %%
########## Parameters ##########
N = 10 # Number of Molecules
################################
# Read the xyz file and then rotate the molecule by euler angles
atoms = sf6.atoms
coords = sf6.geometry
np.random.seed(42)  # For reproducibility

# all_atoms = []
# all_coords = []
pattern = np.zeros((100, 180))  # Initialize pattern array
q_arr = np.linspace(0, 8, 100)      # Radial coordinate (momentum transfer q)
phi_arr = np.linspace(0, 2 * np.pi, 180) # Angular coordinate (azimuthal angle φ)

for i in range(N):
    # Random Euler angles
    rand_alpha = np.random.uniform(0, 360)
    rand_beta = np.random.uniform(0, 360)
    rand_gamma = np.random.uniform(0, 360)
    # Rotate molecule
    rand_rotated_coords = rotate_molecule(coords, rand_alpha, rand_beta, rand_gamma)
    # # Random translation vector, each component in [-10, 10], but ensure norm > 3
    # while True:
    #     translation_vector = np.random.uniform(-10, 10, size=3)
    #     if np.linalg.norm(translation_vector) > 3:
    #         break
    # translated_coords = translate_molecule(rand_rotated_coords, translation_vector)
    # # Append atoms and coordinates for this molecule
    # all_atoms.extend(atoms)
    # all_coords.extend(translated_coords)

    # Define the coordinates for the grid
    write_xyz('sf6_test.xyz', f'{N} random orientations and translations', atoms, np.array(rand_rotated_coords))
    pattern += iam_elastic_pattern_oriented('sf6_test.xyz', q_arr, phi_arr)
pattern /= N  # Average the pattern to the number of molecules
# Create a figure and an axes object with a polar projection
plt.figure(figsize=(15, 5))
plt.yticks([])
plt.xticks([])
plt.box(False)
plt.title('Elastic Scattering Pattern (q·I(q,φ))')
plt.subplot(1,2,1,projection='polar')
# Use pcolormesh to create the plot.
plt.pcolormesh(phi_arr, q_arr, pattern * q_arr[:, np.newaxis], shading='gouraud', cmap='viridis')
plt.grid(False)
plt.xticks([])  # Hide angular tick labels
plt.yticks([])  # Hide radial tick labels
# plt.title('Oriented Elastic Scattering Pattern (q·I(q,φ))')

plt.subplot(1,2,2)
plt.plot(q_arr, pattern.mean(axis=1)*q_arr,'-',label='Oriented Average')
plt.plot(q_arr, iam_elastic_pattern(sf6,q_arr)*q_arr,'--',label='IAM')
plt.legend()
plt.xlabel(r'q ($Å^{-1}$)')
plt.ylabel(r'q·I$_{el}$(q) (arb. units)')
plt.show()

# %%
########## Parameters ##########
N = 1 # Number of Molecules
################################
# Read the xyz file and then rotate the molecule by euler angles
atoms = sf6.atoms
coords = sf6.geometry
np.random.seed(42)  # For reproducibility

# all_atoms = []
# all_coords = []
pattern = np.zeros((100, 180))  # Initialize pattern array
q_arr = np.linspace(0, 8, 100)      # Radial coordinate (momentum transfer q)
phi_arr = np.linspace(0, 2 * np.pi, 180) # Angular coordinate (azimuthal angle φ)

for i in range(N):
    # Random Euler angles
    rand_alpha = np.random.uniform(0, 360)
    rand_beta = np.random.uniform(0, 360)
    rand_gamma = np.random.uniform(0, 360)
    # Rotate molecule
    rand_rotated_coords = rotate_molecule(coords, rand_alpha, rand_beta, rand_gamma)
    # # Random translation vector, each component in [-10, 10], but ensure norm > 3
    # while True:
    #     translation_vector = np.random.uniform(-10, 10, size=3)
    #     if np.linalg.norm(translation_vector) > 3:
    #         break
    # translated_coords = translate_molecule(rand_rotated_coords, translation_vector)
    # # Append atoms and coordinates for this molecule
    # all_atoms.extend(atoms)
    # all_coords.extend(translated_coords)

    # Define the coordinates for the grid
    write_xyz('sf6_test.xyz', f'{N} random orientations and translations', atoms, np.array(rand_rotated_coords))
    pattern += iam_inelastic_pattern_oriented('sf6_test.xyz', q_arr, phi_arr)
pattern /= N  # Average the pattern to the number of molecules
# Create a figure and an axes object with a polar projection
plt.figure(figsize=(15, 5))
plt.yticks([])
plt.xticks([])
plt.box(False)
plt.title('Inelastic Scattering Pattern (q·I(q,φ))')
plt.subplot(1,2,1,projection='polar')
# Use pcolormesh to create the plot.
plt.pcolormesh(phi_arr, q_arr, pattern * q_arr[:, np.newaxis], shading='gouraud', cmap='viridis')
plt.grid(False)
plt.xticks([])  # Hide angular tick labels
plt.yticks([])  # Hide radial tick labels
# plt.title('Oriented Elastic Scattering Pattern (q·I(q,φ))')

plt.subplot(1,2,2)
plt.plot(q_arr, pattern.mean(axis=1)*q_arr,'-',label='Oriented Average')
plt.plot(q_arr, iam_inelastic_pattern(sf6,q_arr)*q_arr,'--',label='IAM')
plt.legend()
plt.xlabel(r'q ($Å^{-1}$)')
plt.ylabel(r'q·I$_{inel}$(q) (arb. units)')
plt.show()
