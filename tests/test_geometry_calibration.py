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
from xrayscatteringtools.calib.scattering_corrections import correction_factor
from xrayscatteringtools.utils import theta2q


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

    @pytest.mark.parametrize("z0", [40_000.0, 90_000.0])
    @pytest.mark.parametrize("phi0", [0.0, 0.3, -1.2, np.pi / 2])
    def test_matches_trig_formula(self, z0, phi0):
        """The trig-free form equals sin^2(phi) + cos^2(theta) cos^2(phi) from Ma et al."""
        rng = np.random.default_rng(1)
        x = rng.uniform(-60_000, 60_000, 1000)
        y = rng.uniform(-60_000, 60_000, 1000)
        theta = np.arctan(np.sqrt(x**2 + y**2) / z0)
        phi = np.arctan2(y, x) + phi0
        expected = np.sin(phi) ** 2 + np.cos(theta) ** 2 * np.cos(phi) ** 2
        np.testing.assert_allclose(thompson_correction(x, y, z0, phi0), expected, atol=1e-14)


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

    def test_matches_trig_formula(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(-60_000, 60_000, 1000)
        y = rng.uniform(-60_000, 60_000, 1000)
        expected = np.cos(np.arctan(np.sqrt(x**2 + y**2) / 90_000)) ** 3
        np.testing.assert_allclose(geometry_correction(x, y, 90_000), expected, rtol=1e-14)


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

    def test_matches_product_of_corrections(self, peaked_theory):
        """model = amplitude * I(q) * thompson * geometry * correction_factor."""
        x, y = _make_detector(size=16, pixel_pitch=2000.0)
        x0, y0, z0, phi0, keV = 300.0, -200.0, 60_000.0, 0.2, 12.0
        cx, cy = x.ravel() - x0, y.ravel() - y0
        q = theta2q(np.arctan(np.sqrt(cx**2 + cy**2) / z0), keV)
        expected = (2.0 * peaked_theory(q) * thompson_correction(cx, cy, z0, phi0)
                    * geometry_correction(cx, cy, z0) * correction_factor(q, keV))
        result = model([x.ravel(), y.ravel()], 2.0, x0, y0, z0, phi0, keV, peaked_theory)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_background_is_unpolarized_and_corrected(self, peaked_theory):
        """The background adds amplitude * background * geometry * correction_factor, with no Thompson factor."""
        x, y = _make_detector(size=16, pixel_pitch=2000.0)
        xy = [x.ravel(), y.ravel()]
        z0, keV = 60_000.0, 12.0
        with_bg = model(xy, 2.0, 0, 0, z0, 0, keV, peaked_theory, background=3.0)
        without = model(xy, 2.0, 0, 0, z0, 0, keV, peaked_theory)
        q = theta2q(np.arctan(np.sqrt(xy[0]**2 + xy[1]**2) / z0), keV)
        expected = 2.0 * 3.0 * geometry_correction(xy[0], xy[1], z0) * correction_factor(q, keV)
        np.testing.assert_allclose(with_bg - without, expected, rtol=1e-10)

    def test_zero_background_is_default(self, peaked_theory):
        x, y = _make_detector(size=8)
        xy = [x.ravel(), y.ravel()]
        np.testing.assert_array_equal(
            model(xy, 1.0, 0, 0, 90_000, 0, 10.0, peaked_theory, background=0.0),
            model(xy, 1.0, 0, 0, 90_000, 0, 10.0, peaked_theory),
        )


# ── run_geometry_calibration ───────────────────────────────────────────


def _synthetic_image(background_true=0.0, size=64, pixel_pitch=75.0, z0=90_000.0):
    """Synthetic raw image from the model with known geometry."""
    x, y = _make_detector(size=size, pixel_pitch=pixel_pitch)
    q_theory = np.linspace(0, 15, 300)
    Iq_theory = 100 * np.exp(-q_theory ** 2 / 8)
    theory_interp = InterpolatedUnivariateSpline(q_theory, Iq_theory, ext=3)
    raw = model(
        [x.ravel(), y.ravel()],
        50.0, 200.0, -150.0, z0,
        0, 10.0, theory_interp, background=background_true,
    ).reshape(x.shape)
    return x, y, raw, q_theory, Iq_theory


class TestRunGeometryCalibration:
    def test_recovers_known_geometry(self):
        """Fit synthetic data generated from model and verify parameters are recovered."""
        x, y, raw, q_theory, Iq_theory = _synthetic_image()
        mask = np.ones_like(raw, dtype=bool)

        fit, popt, pcov = run_geometry_calibration(
            raw, x, y, mask,
            q_theory, Iq_theory, 10.0,
            initial_guess={'amplitude': 40, 'x0': 0, 'y0': 0, 'z0': 85_000},
            mask_center=False,
        )

        # popt = [amplitude, x0, y0, z0]
        assert popt[0] == pytest.approx(50.0, rel=0.05)
        assert popt[1] == pytest.approx(200.0, abs=200)
        assert popt[2] == pytest.approx(-150.0, abs=200)
        assert popt[3] == pytest.approx(90_000.0, rel=0.02)

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

    def test_result_independent_of_threads(self):
        """Splitting the model over threads gives exactly the same fit."""
        x, y, raw, q_theory, Iq_theory = _synthetic_image(size=32)
        mask = np.ones_like(raw, dtype=bool)
        kwargs = dict(initial_guess={'amplitude': 40, 'x0': 0, 'y0': 0, 'z0': 85_000}, mask_center=False)
        fit1, popt1, _ = run_geometry_calibration(raw, x, y, mask, q_theory, Iq_theory, 10.0, n_threads=1, **kwargs)
        fit3, popt3, _ = run_geometry_calibration(raw, x, y, mask, q_theory, Iq_theory, 10.0, n_threads=3, **kwargs)
        np.testing.assert_array_equal(popt1, popt3)
        np.testing.assert_array_equal(fit1, fit3)


class TestRunGeometryCalibrationBackground:
    # Large pixels and a short distance so the pattern reaches high q, where the
    # background is distinct from the amplitude.
    _image_kwargs = dict(size=96, pixel_pitch=300.0, z0=40_000.0)
    _guess = {'amplitude': 40, 'x0': 0, 'y0': 0, 'z0': 37_000}

    def test_recovers_geometry_and_background(self):
        x, y, raw, q_theory, Iq_theory = _synthetic_image(background_true=2.0, **self._image_kwargs)
        mask = np.ones_like(raw, dtype=bool)
        fit, popt, pcov = run_geometry_calibration(
            raw, x, y, mask, q_theory, Iq_theory, 10.0,
            initial_guess=self._guess, mask_center=False, fit_background=True,
        )
        assert len(popt) == 5
        assert pcov.shape == (5, 5)
        assert popt[0] == pytest.approx(50.0, rel=0.02)
        assert popt[1] == pytest.approx(200.0, abs=100)
        assert popt[2] == pytest.approx(-150.0, abs=100)
        assert popt[3] == pytest.approx(40_000.0, rel=0.02)
        assert popt[4] == pytest.approx(2.0, rel=0.05)
        np.testing.assert_allclose(fit, raw, rtol=1e-3)

    def test_no_background_fits_zero(self):
        x, y, raw, q_theory, Iq_theory = _synthetic_image(background_true=0.0, **self._image_kwargs)
        mask = np.ones_like(raw, dtype=bool)
        _, popt, _ = run_geometry_calibration(
            raw, x, y, mask, q_theory, Iq_theory, 10.0,
            initial_guess={**self._guess, 'background': 1}, mask_center=False, fit_background=True,
        )
        assert popt[4] == pytest.approx(0.0, abs=0.05)

    def test_accepts_five_bounds(self):
        x, y, raw, q_theory, Iq_theory = _synthetic_image(background_true=2.0, **self._image_kwargs)
        mask = np.ones_like(raw, dtype=bool)
        bounds = ([0, -10_000, -10_000, 30_000, 0], [np.inf, 10_000, 10_000, 150_000, 10])
        _, popt, _ = run_geometry_calibration(
            raw, x, y, mask, q_theory, Iq_theory, 10.0,
            initial_guess=self._guess, mask_center=False, fit_background=True, bounds=bounds,
        )
        assert popt[4] == pytest.approx(2.0, rel=0.05)

    def test_off_by_default(self):
        x, y, raw, q_theory, Iq_theory = _synthetic_image(**self._image_kwargs)
        mask = np.ones_like(raw, dtype=bool)
        _, popt, pcov = run_geometry_calibration(
            raw, x, y, mask, q_theory, Iq_theory, 10.0,
            initial_guess={**self._guess, 'background': 5}, mask_center=False,
        )
        assert len(popt) == 4
        assert pcov.shape == (4, 4)
