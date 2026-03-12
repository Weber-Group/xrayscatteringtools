# %%
from xrayscatteringtools.theory.patterns import SF6__MP2__aug_cc_pVDZ as MP2
from xrayscatteringtools.theory.patterns import SF6__CCSD__aug_cc_pVDZ as CCSD
from xrayscatteringtools.theory.patterns import SF6__HF__aug_cc_pVDZ as HF
from xrayscatteringtools.theory.geometries import SF6__CCSD_T_DHK__aug_cc_pV5Z_DK as sf6
from xrayscatteringtools.theory import iam_total_pattern, iam_inelastic_pattern, iam_elastic_pattern
import matplotlib.pyplot as plt

# %% [markdown]
# ### Elastic Differences

# %% [markdown]
# ### Total Comparisons

# %%
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(MP2.q, MP2.I_q_elastic*MP2.q, label="MP2 Elastic")
plt.plot(CCSD.q, CCSD.I_q_elastic*CCSD.q, label="CCSD Elastic")
plt.plot(HF.q, HF.I_q_elastic*HF.q, label="HF Elastic")
plt.plot(MP2.q, iam_elastic_pattern(sf6, MP2.q)*MP2.q, label="IAM Elastic")
plt.xlabel("Momentum Transfer q (Å⁻¹)")
plt.ylabel("q * I(q) (arb. units)")
plt.title("Elastic Scattering Intensity Comparison")
plt.legend()
plt.subplot(1, 2, 2)
# Difference Plots
plt.plot(MP2.q, (MP2.I_q_elastic - CCSD.I_q_elastic), label="MP2 - CCSD")
plt.plot(MP2.q, (MP2.I_q_elastic - iam_elastic_pattern(sf6, MP2.q)), label="MP2 - IAM")
plt.plot(CCSD.q, (CCSD.I_q_elastic - iam_elastic_pattern(sf6, CCSD.q)), label="CCSD - IAM")
plt.xlabel("Momentum Transfer q (Å⁻¹)")
plt.ylabel("q * ΔI(q) (arb. units)")
plt.title("Elastic Scattering Intensity Differences")
plt.legend()
plt.show()

# %% [markdown]
# ### Inelastic differences

# %%
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(MP2.q, MP2.I_q_inelastic, label="MP2 Inelastic")
plt.plot(CCSD.q, CCSD.I_q_inelastic, label="CCSD Inelastic")
plt.plot(HF.q, HF.I_q_inelastic, label="HF Inelastic", linestyle='dashed')
plt.plot(MP2.q, iam_inelastic_pattern(sf6, MP2.q), label="IAM Inelastic")
plt.xlabel("Momentum Transfer q (Å⁻¹)")
plt.ylabel("I(q) (arb. units)")
plt.title("Inelastic Scattering Intensity Comparison")
plt.legend()
plt.xlim(0,4.5)
plt.subplot(1, 2, 2)
# Difference Plots
plt.plot(MP2.q, (MP2.I_q_inelastic - CCSD.I_q_inelastic), label="MP2 - CCSD")
plt.plot(MP2.q, (MP2.I_q_inelastic - iam_inelastic_pattern(sf6, MP2.q)), label="MP2 - IAM")
plt.plot(CCSD.q, (CCSD.I_q_inelastic - iam_inelastic_pattern(sf6, CCSD.q)), label="CCSD - IAM")
plt.xlabel("Momentum Transfer q (Å⁻¹)")
plt.ylabel("q * ΔI(q) (arb. units)")
plt.title("Inelastic Scattering Intensity Differences")
plt.legend()
plt.show()

# %%
# plt.plot(MP2.q, MP2.I_q_elastic/MP2.I_q, label="MP2 Elastic Ratio")
plt.plot(CCSD.q, (CCSD.I_q_elastic)/CCSD.I_q, label="CCSD Elastic Ratio")
plt.plot(CCSD.q, (CCSD.I_q-(CCSD.I_q_inelastic-5))/CCSD.I_q, label="CCSD Elastic Ratio Offset")
# plt.plot(HF.q, HF.I_q_elastic/HF.I_q, label="HF Elastic Ratio")
# plt.plot(MP2.q, iam_elastic_pattern(sf6, MP2.q)/iam_total_pattern(sf6, MP2.q), label="IAM Elastic Ratio")
plt.axhline(1, color='black', linestyle='dashed')
plt.xlim(0,4.2)
plt.legend()

# %%
from xrayscatteringtools.utils import compute_G_of_r
r, G = compute_G_of_r(CCSD.q[0:60], (CCSD.I_q-70)[0:60], r_max=5.0)
r, GHF = compute_G_of_r(CCSD.q[0:60], (CCSD.I_q_elastic)[0:60], r_max=5.0)

# %%
r, G = compute_G_of_r(CCSD.q, (CCSD.I_q-70), r_max=5.0)
r, GHF = compute_G_of_r(CCSD.q, (CCSD.I_q_elastic), r_max=5.0)

# %%
plt.plot(r,G-GHF)
