"""Tests for xrayscatteringtools.calib.scattering_corrections."""

import numpy as np
import pytest

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
