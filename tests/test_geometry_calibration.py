"""Tests for xrayscatteringtools.calib.geometry_calibration."""

import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline

from xrayscatteringtools.calib.geometry_calibration import (
    thompson_correction,
    geometry_correction,
    geometry_correction_units,
    model,
    run_geometry_calibration,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _flat_theory(q):
    """A trivial flat theory pattern for testing: I(q) = 1 everywhere."""
    return np.ones_like(q)


def _make_detector(size=32, pixel_pitch=75.0, z0=90_000):
    """Create a small synthetic detector grid centered at (0,0)."""
    half = size // 2
    coords = np.arange(-half, half) * pixel_pitch
    y2d, x2d = np.meshgrid(coords, coords)
    return x2d, y2d


# ── thompson_correction ───────────────────────────────────────────────


class TestThompsonCorrection:
    def test_on_axis_returns_one(self):
        """At (0,0) theta=0, correction should be 1 regardless of phi0."""
        x = np.array([0.0])
        y = np.array([0.0])
        result = thompson_correction(x, y, z0=90_000, phi0=0)
        assert result.item() == pytest.approx(1.0)

    def test_shape_preserved(self):
        x, y = _make_detector()
        result = thompson_correction(x, y, z0=90_000, phi0=0)
        assert result.shape == x.shape

    def test_values_in_valid_range(self):
        """Thompson correction should be between 0 and 1."""
        x, y = _make_detector()
        result = thompson_correction(x, y, z0=90_000, phi0=0)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0 + 1e-12)

    def test_symmetric_about_origin(self):
        """Symmetric pixels should yield the same correction for phi0=0."""
        x = np.array([1000.0, -1000.0])
        y = np.array([0.0, 0.0])
        result = thompson_correction(x, y, z0=90_000, phi0=0)
        assert result[0] == pytest.approx(result[1])

    def test_phi0_rotation(self):
        """Rotating phi0 by pi/2 should swap x/y roles."""
        x = np.array([5000.0])
        y = np.array([0.0])
        c1 = thompson_correction(x, y, z0=90_000, phi0=0)
        c2 = thompson_correction(x, y, z0=90_000, phi0=np.pi / 2)
        # They should differ (unless theta is exactly 0)
        assert not np.isclose(c1, c2)


# ── geometry_correction ────────────────────────────────────────────────


class TestGeometryCorrection:
    def test_on_axis_returns_one(self):
        """At (0,0), theta=0, cos^3(0)=1."""
        result = geometry_correction(np.array([0.0]), np.array([0.0]), z0=90_000)
        assert result.item() == pytest.approx(1.0)

    def test_shape_preserved(self):
        x, y = _make_detector()
        result = geometry_correction(x, y, z0=90_000)
        assert result.shape == x.shape

    def test_decreases_off_axis(self):
        """Correction should decrease as we move further from center."""
        x = np.array([0.0, 10_000.0, 50_000.0])
        y = np.zeros(3)
        result = geometry_correction(x, y, z0=90_000)
        assert result[0] > result[1] > result[2]

    def test_bounded_zero_to_one(self):
        x, y = _make_detector()
        result = geometry_correction(x, y, z0=90_000)
        assert np.all(result > 0)
        assert np.all(result <= 1.0 + 1e-12)

    def test_symmetric(self):
        x = np.array([5000.0, -5000.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 5000.0, -5000.0])
        result = geometry_correction(x, y, z0=90_000)
        np.testing.assert_allclose(result, result[0])

    def test_known_value(self):
        """cos^3(arctan(r/z0)) for r=z0 -> cos^3(pi/4) = (1/sqrt(2))^3."""
        z0 = 90_000.0
        result = geometry_correction(np.array([z0]), np.array([0.0]), z0=z0)
        expected = (1.0 / np.sqrt(2)) ** 3
        assert result.item() == pytest.approx(expected)


# ── geometry_correction_units ──────────────────────────────────────────


class TestGeometryCorrectionUnits:
    def test_on_axis_value(self):
        """At center: cos^3(0) * dx*dy / z0^2 = dx*dy / z0^2."""
        z0 = 90_000.0
        dx, dy = 75.0, 75.0
        result = geometry_correction_units(
            np.array([0.0]), np.array([0.0]), z0, dx, dy
        )
        expected = dx * dy / z0 ** 2
        assert result.item() == pytest.approx(expected)

    def test_shape_preserved(self):
        x, y = _make_detector()
        result = geometry_correction_units(x, y, z0=90_000, dx=75, dy=75)
        assert result.shape == x.shape

    def test_proportional_to_pixel_area(self):
        """Doubling pixel size should quadruple the correction."""
        x = np.array([10_000.0])
        y = np.array([0.0])
        c1 = geometry_correction_units(x, y, z0=90_000, dx=75, dy=75)
        c2 = geometry_correction_units(x, y, z0=90_000, dx=150, dy=150)
        assert c2.item() == pytest.approx(4 * c1.item())


# ── model ──────────────────────────────────────────────────────────────


class TestModel:
    @pytest.fixture
    def flat_theory(self):
        """A flat I(q)=1 spline for isolating correction behavior."""
        q = np.linspace(0, 20, 200)
        return InterpolatedUnivariateSpline(q, np.ones_like(q), ext=3)

    @pytest.fixture
    def peaked_theory(self):
        """A Gaussian-like theory pattern."""
        q = np.linspace(0, 20, 200)
        Iq = 100 * np.exp(-q ** 2 / 10)
        return InterpolatedUnivariateSpline(q, Iq, ext=3)

    def test_output_length(self, flat_theory):
        x, y = _make_detector(size=16)
        xy = [x.ravel(), y.ravel()]
        result = model(xy, 1.0, 0, 0, 90_000, 0, 10.0, flat_theory)
        assert result.shape == (16 * 16,)

    def test_amplitude_scaling(self, flat_theory):
        x, y = _make_detector(size=8)
        xy = [x.ravel(), y.ravel()]
        r1 = model(xy, 1.0, 0, 0, 90_000, 0, 10.0, flat_theory)
        r3 = model(xy, 3.0, 0, 0, 90_000, 0, 10.0, flat_theory)
        np.testing.assert_allclose(r3, 3 * r1, rtol=1e-12)

    def test_no_corrections(self, flat_theory):
        """With all corrections off and flat theory, result = amplitude everywhere."""
        x, y = _make_detector(size=8)
        xy = [x.ravel(), y.ravel()]
        result = model(
            xy, 5.0, 0, 0, 90_000, 0, 10.0, flat_theory,
            do_geometry_correction=False,
            do_thompson_correction=False,
            do_angle_of_scattering_correction=False,
        )
        np.testing.assert_allclose(result, 5.0, rtol=1e-10)

    def test_shift_center(self, peaked_theory):
        """Shifting x0/y0 should move the pattern center."""
        x, y = _make_detector(size=16)
        xy = [x.ravel(), y.ravel()]
        r_centered = model(
            xy, 1.0, 0, 0, 90_000, 0, 10.0, peaked_theory,
            do_geometry_correction=False,
            do_thompson_correction=False,
            do_angle_of_scattering_correction=False,
        )
        r_shifted = model(
            xy, 1.0, 500, 0, 90_000, 0, 10.0, peaked_theory,
            do_geometry_correction=False,
            do_thompson_correction=False,
            do_angle_of_scattering_correction=False,
        )
        # The two patterns should differ
        assert not np.allclose(r_centered, r_shifted)

    def test_positive_output(self, peaked_theory):
        x, y = _make_detector(size=16)
        xy = [x.ravel(), y.ravel()]
        result = model(xy, 1.0, 0, 0, 90_000, 0, 10.0, peaked_theory)
        assert np.all(result >= 0)


# ── run_geometry_calibration ───────────────────────────────────────────


class TestRunGeometryCalibration:
    def test_recovers_known_geometry(self):
        """Fit synthetic data generated from model and verify parameters are recovered."""
        size = 64
        x, y = _make_detector(size=size, pixel_pitch=75.0)
        z0_true = 90_000.0
        x0_true = 200.0
        y0_true = -150.0
        amp_true = 50.0
        keV = 10.0

        # Build a simple theory pattern
        q_theory = np.linspace(0, 15, 300)
        Iq_theory = 100 * np.exp(-q_theory ** 2 / 8)
        theory_interp = InterpolatedUnivariateSpline(q_theory, Iq_theory, ext=3)

        # Generate a synthetic "raw image" from the model
        raw = model(
            [x.ravel(), y.ravel()],
            amp_true, x0_true, y0_true, z0_true,
            0, keV, theory_interp,
        ).reshape(x.shape)

        mask = np.ones_like(raw, dtype=bool)

        fit, popt, pcov = run_geometry_calibration(
            raw, x, y, mask,
            q_theory, Iq_theory, keV,
            initial_guess={'amplitude': 40, 'x0': 0, 'y0': 0, 'z0': 85_000},
            mask_center=False,
        )

        # popt = [amplitude, x0, y0, z0]
        assert popt[0] == pytest.approx(amp_true, rel=0.05)
        assert popt[1] == pytest.approx(x0_true, abs=200)
        assert popt[2] == pytest.approx(y0_true, abs=200)
        assert popt[3] == pytest.approx(z0_true, rel=0.02)

    def test_output_shapes(self):
        """Basic shape checks on returned fit, popt, pcov."""
        size = 16
        x, y = _make_detector(size=size)
        raw = np.ones_like(x, dtype=float)
        mask = np.ones_like(raw, dtype=bool)

        q_theory = np.linspace(0, 15, 100)
        Iq_theory = np.ones_like(q_theory)

        fit, popt, pcov = run_geometry_calibration(
            raw, x, y, mask,
            q_theory, Iq_theory, 10.0,
            mask_center=False,
        )
        assert fit.shape == raw.shape
        assert len(popt) == 4
        assert pcov.shape == (4, 4)
