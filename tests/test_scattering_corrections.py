"""Tests for xrayscatteringtools.calib.scattering_corrections."""

import inspect

import numpy as np
import pytest
import xraylib

from xrayscatteringtools.calib.scattering_corrections import (
    correction_factor,
    Si_correction,
    KaptonHN_correction,
    Al_correction,
    Be_correction,
    cell_correction,
    Si_attenuation_length,
    Al_attenuation_length,
    Be_attenuation_length,
    KaptonHN_attenuation_length,
    Zn_attenuation_length,
    J4M_efficiency,
)


# ── Attenuation lengths ───────────────────────────────────────────────


class TestAttenuationLengths:
    """Each attenuation-length function should return a positive value in meters."""

    @pytest.mark.parametrize("func", [
        Si_attenuation_length,
        Al_attenuation_length,
        Be_attenuation_length,
        KaptonHN_attenuation_length,
        Zn_attenuation_length,
    ])
    def test_positive_at_10keV(self, func):
        result = func(10.0)
        assert np.isfinite(result)
        assert result > 0

    @pytest.mark.parametrize("func", [
        Si_attenuation_length,
        Al_attenuation_length,
        Be_attenuation_length,
        KaptonHN_attenuation_length,
        Zn_attenuation_length,
    ])
    def test_returns_scalar(self, func):
        result = func(9.5)
        assert np.ndim(result) == 0

    @pytest.mark.parametrize("func", [
        Si_attenuation_length,
        Al_attenuation_length,
        Be_attenuation_length,
        KaptonHN_attenuation_length,
        Zn_attenuation_length,
    ])
    def test_accepts_arrays_and_ints(self, func):
        energies = np.array([8, 10, 12])
        result = func(energies)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [func(8.0), func(10.0), func(12.0)])

    def test_Si_longer_than_Zn(self):
        """At the same energy, lighter materials generally have longer attenuation lengths."""
        assert Si_attenuation_length(10.0) > Zn_attenuation_length(10.0)

    def test_Be_longer_than_Si(self):
        """Be is lighter than Si — expect longer attenuation length."""
        assert Be_attenuation_length(10.0) > Si_attenuation_length(10.0)

    @pytest.mark.parametrize("func", [
        Si_attenuation_length,
        Al_attenuation_length,
        Be_attenuation_length,
        KaptonHN_attenuation_length,
    ])
    def test_increases_with_energy(self, func):
        """Attenuation length generally increases with photon energy (away from edges)."""
        low = func(8.0)
        high = func(12.0)
        assert high > low

    @pytest.mark.parametrize("func, cross_section, material, density", [
        (Si_attenuation_length, xraylib.CS_Photo, 14, 2.33),
        (Al_attenuation_length, xraylib.CS_Total, 13, 2.699),
        (Be_attenuation_length, xraylib.CS_Total, 4, 1.848),
        (KaptonHN_attenuation_length, xraylib.CS_Total_CP, "C22H10N2O5", 1.42),
        (Zn_attenuation_length, xraylib.CS_Total, 30, 7.14),
    ])
    def test_matches_xraylib(self, func, cross_section, material, density):
        """1 / (mass attenuation * density), converted from cm to m."""
        expected = 1e-2 / (cross_section(material, 15.0) * density)
        assert func(15.0) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("func, keV, henke_um", [
        (Si_attenuation_length, 10.0, 133.72),
        (Al_attenuation_length, 10.0, 149.63),
        (Zn_attenuation_length, 8.0, 24.48),
    ])
    def test_close_to_henke(self, func, keV, henke_um):
        """Mid- and high-Z materials agree with the Henke (CXRO) tables to within a few percent."""
        assert func(keV) * 1e6 == pytest.approx(henke_um, rel=0.06)


# ── Individual material corrections ───────────────────────────────────


class TestSiCorrection:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = Si_correction(q, keV=10.0)
        assert result.shape == q.shape

    def test_near_unity_at_small_q(self):
        """At small scattering angles, correction should be close to 1."""
        q = np.array([0.01])
        result = Si_correction(q, keV=10.0)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_finite(self):
        q = np.linspace(0.5, 10, 50)
        result = Si_correction(q, keV=10.0)
        assert np.all(np.isfinite(result))


class TestKaptonHNCorrection:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = KaptonHN_correction(q, keV=10.0)
        assert result.shape == q.shape

    def test_near_unity_at_small_q(self):
        q = np.array([0.01])
        result = KaptonHN_correction(q, keV=10.0)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_positive(self):
        q = np.linspace(0.5, 10, 50)
        result = KaptonHN_correction(q, keV=10.0)
        assert np.all(result > 0)


class TestAlCorrection:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = Al_correction(q, keV=10.0)
        assert result.shape == q.shape

    def test_near_unity_at_small_q(self):
        q = np.array([0.01])
        result = Al_correction(q, keV=10.0)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_positive(self):
        q = np.linspace(0.5, 10, 50)
        result = Al_correction(q, keV=10.0)
        assert np.all(result > 0)


class TestBeCorrection:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = Be_correction(q, keV=10.0)
        assert result.shape == q.shape

    def test_positive(self):
        q = np.linspace(0.5, 10, 50)
        result = Be_correction(q, keV=10.0)
        assert np.all(result > 0)

    def test_finite(self):
        q = np.linspace(0.1, 10, 100)
        result = Be_correction(q, keV=10.0)
        assert np.all(np.isfinite(result))

    def test_one_at_zero_q(self):
        """On axis the photon leaves through the hole, so there is no Be attenuation."""
        assert Be_correction(np.array([0.0]), keV=10.0).item() == pytest.approx(1.0)


class TestCellCorrection:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = cell_correction(q, keV=10.0)
        assert result.shape == q.shape

    def test_finite(self):
        q = np.linspace(0.1, 10, 100)
        result = cell_correction(q, keV=10.0)
        assert np.all(np.isfinite(result))

    def test_positive(self):
        q = np.linspace(0.5, 10, 50)
        result = cell_correction(q, keV=10.0)
        assert np.all(result > 0)

    def test_at_least_one(self):
        """Cell correction should be >= 1 (adds signal from cell geometry)."""
        q = np.linspace(0.5, 8, 50)
        result = cell_correction(q, keV=10.0)
        assert np.all(result >= 1.0 - 1e-10)

    def test_accepts_list(self):
        result = cell_correction([0.5, 1.0, 2.0], keV=10.0)
        assert result.shape == (3,)


# ── Pinhole defaults ──────────────────────────────────────────────────


class TestPinholeDefaults:
    """Both pinholes default to a 250 µm diameter (125 µm radius)."""

    @pytest.mark.parametrize("func, arg", [
        (correction_factor, "rBe"),
        (correction_factor, "rP"),
        (Be_correction, "rBe"),
        (cell_correction, "rP"),
    ])
    def test_radius_is_125_um(self, func, arg):
        assert inspect.signature(func).parameters[arg].default == 125e-6


# ── correction_factor (total) ─────────────────────────────────────────


class TestCorrectionFactor:
    def test_shape(self):
        q = np.linspace(0.5, 10, 50)
        result = correction_factor(q, keV=10.0)
        assert result.shape == q.shape

    def test_finite(self):
        q = np.linspace(0.5, 10, 50)
        result = correction_factor(q, keV=10.0)
        assert np.all(np.isfinite(result))

    def test_positive_at_moderate_q(self):
        """At moderate q the total correction should be positive."""
        q = np.linspace(0.5, 5, 30)
        result = correction_factor(q, keV=10.0)
        assert np.all(result > 0)

    def test_is_product_of_parts(self):
        """Total correction should equal the product of individual corrections."""
        q = np.linspace(0.5, 8, 40)
        keV = 10.0
        total = correction_factor(q, keV)
        manual = (
            Si_correction(q, keV)
            * KaptonHN_correction(q, keV)
            * Al_correction(q, keV)
            * Be_correction(q, keV)
            * cell_correction(q, keV)
        )
        np.testing.assert_allclose(total, manual, rtol=1e-12)

    def test_negative_keV_raises(self):
        with pytest.raises(ValueError, match="positive"):
            correction_factor(np.array([1.0]), keV=-5)

    def test_zero_keV_raises(self):
        with pytest.raises(ValueError, match="positive"):
            correction_factor(np.array([1.0]), keV=0)

    def test_custom_thicknesses(self):
        """Should accept and respect custom material thicknesses."""
        q = np.linspace(1, 5, 20)
        c_default = correction_factor(q, keV=10.0)
        c_thick = correction_factor(q, keV=10.0, tSi=600e-6)
        # Thicker Si should change the correction
        assert not np.allclose(c_default, c_thick)

    def test_scalar_q(self):
        result = correction_factor(np.array([2.0]), keV=10.0)
        assert result.shape == (1,)
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("keV", [2.0, 25.0])
    def test_works_outside_old_table_range(self, keV):
        """xraylib covers energies the old 6-20 keV tables did not."""
        q = np.linspace(0.1, 1.0, 10)
        assert np.all(np.isfinite(correction_factor(q, keV)))


# ── J4M_efficiency ────────────────────────────────────────────────────


class TestJ4MEfficiency:
    def test_normal_incidence(self):
        """At theta=0 (normal incidence), efficiency should be between 0 and 1."""
        eff = J4M_efficiency(0.0, 10.0)
        assert 0 < eff < 1

    def test_bounded(self):
        """Efficiency should always be between 0 and 1."""
        thetas = np.linspace(0, 0.5, 50)
        eff = J4M_efficiency(thetas, 10.0)
        assert np.all(eff >= 0)
        assert np.all(eff <= 1.0 + 1e-12)

    def test_varies_with_angle(self):
        """Efficiency should change with scattering angle."""
        eff_0 = J4M_efficiency(0.0, 10.0)
        eff_high = J4M_efficiency(0.3, 10.0)
        assert eff_0 != pytest.approx(eff_high)

    def test_shape(self):
        thetas = np.linspace(0, 0.4, 30)
        eff = J4M_efficiency(thetas, 10.0)
        assert eff.shape == thetas.shape

    def test_custom_thicknesses(self):
        """Thicker Si should increase efficiency (more absorption in sensor)."""
        eff_default = J4M_efficiency(0.0, 10.0, tSi=318.5e-6)
        eff_thick = J4M_efficiency(0.0, 10.0, tSi=1000e-6)
        assert eff_thick > eff_default

    def test_known_value_at_10keV(self):
        """About 88% at 10 keV on axis (the old Henke tables gave 0.878)."""
        assert J4M_efficiency(0.0, 10.0) == pytest.approx(0.883, abs=0.005)
