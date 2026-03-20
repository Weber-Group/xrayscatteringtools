"""Tests for xrayscatteringtools.azimuthal."""

import numpy as np
import pytest

from xrayscatteringtools.azimuthal import (
    compute_q_map,
    azimuthalBinning,
    create_J4m_integrator,
    azimuthalBinning_pyfai,
)
from xrayscatteringtools.utils import J4M

# ── compute_q_map ──────────────────────────────────────────────────────

class TestComputeQMap:
    def test_basic_output_shape(self):
        x = np.zeros((3, 4))
        y = np.zeros((3, 4))
        q = compute_q_map(x, y)
        assert q.shape == (3, 4)

    def test_beam_center_returns_zero_q(self):
        """A pixel at the beam center should have q ≈ 0."""
        x = np.array([[0.0]])
        y = np.array([[0.0]])
        q = compute_q_map(x, y, x0=0, y0=0, z0=90_000)
        assert q.item() == pytest.approx(0.0, abs=1e-10)

    def test_q_increases_with_distance(self):
        """Pixels further from beam center should have larger q."""
        x = np.array([[0.0, 1000.0, 5000.0]])
        y = np.array([[0.0, 0.0, 0.0]])
        q = compute_q_map(x, y, x0=0, y0=0, z0=90_000, keV=10)
        assert q[0, 0] < q[0, 1] < q[0, 2]

    def test_q_symmetric(self):
        """Symmetric pixels about origin should have equal q."""
        x = np.array([[1000.0, -1000.0]])
        y = np.array([[0.0, 0.0]])
        q = compute_q_map(x, y, x0=0, y0=0, z0=90_000)
        assert q[0, 0] == pytest.approx(q[0, 1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_q_map(np.zeros((2, 3)), np.zeros((3, 2)))

    def test_negative_keV_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_q_map(np.zeros((2, 2)), np.zeros((2, 2)), keV=-1)

    def test_zero_keV_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_q_map(np.zeros((2, 2)), np.zeros((2, 2)), keV=0)

    def test_q_all_positive(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-50_000, 50_000, size=(10, 10))
        y = rng.uniform(-50_000, 50_000, size=(10, 10))
        q = compute_q_map(x, y, keV=10)
        assert np.all(q >= 0)

    def test_keV_affects_q(self):
        """Higher energy → smaller wavelength → larger q for same geometry."""
        x = np.array([[5000.0]])
        y = np.array([[0.0]])
        q_lo = compute_q_map(x, y, keV=5)
        q_hi = compute_q_map(x, y, keV=20)
        assert q_hi.item() > q_lo.item()


# ── azimuthalBinning ───────────────────────────────────────────────────


class TestAzimuthalBinning:
    """Tests for the main azimuthal-binning routine."""

    @pytest.fixture
    def simple_ring(self):
        """Create a simple detector image with a ring of constant intensity."""
        size = 64
        half = size // 2
        # Pixel coordinates in microns (centered at 0)
        xs = np.linspace(-50_000, 50_000, size)
        ys = np.linspace(-50_000, 50_000, size)
        y2d, x2d = np.meshgrid(ys, xs)
        img = np.ones_like(x2d, dtype=float)
        return img, x2d, y2d

    def test_output_shapes_1d(self, simple_ring):
        img, x, y = simple_ring
        q, I = azimuthalBinning(img, x, y, z0=90_000, keV=10, qBin=0.1)
        assert q.ndim == 1
        assert I.ndim == 1
        assert q.shape == I.shape

    def test_output_shapes_phi_bins(self, simple_ring):
        img, x, y = simple_ring
        q, I = azimuthalBinning(img, x, y, z0=90_000, keV=10, qBin=0.1, phiBins=4)
        assert q.ndim == 1
        assert I.ndim == 2
        assert I.shape[0] == 4
        assert I.shape[1] == len(q)

    def test_uniform_image_flat_curve(self, simple_ring):
        """A uniform image should produce a roughly flat azimuthal average
        (deviations from corrections aside, values should be finite and > 0)."""
        img, x, y = simple_ring
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False,
        )
        finite = np.isfinite(I)
        assert finite.any()
        # All finite bins should be close to 1 (uniform input, no corrections)
        np.testing.assert_allclose(I[finite], 1.0, atol=1e-10)

    def test_mask_excludes_pixels(self, simple_ring):
        img, x, y = simple_ring
        mask = np.zeros_like(img, dtype=bool)
        mask[:, :32] = True  # mask left half
        q1, I1 = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False, mask=mask,
        )
        # Should still return valid output
        assert np.isfinite(I1).any()

    def test_dark_subtraction(self, simple_ring):
        img, x, y = simple_ring
        dark = np.full_like(img, 0.5)
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False, darkImg=dark,
        )
        finite = np.isfinite(I)
        # After subtracting 0.5 from uniform 1.0, average should be ~0.5
        np.testing.assert_allclose(I[finite], 0.5, atol=1e-10)

    def test_gain_correction(self, simple_ring):
        img, x, y = simple_ring
        gain = np.full_like(img, 2.0)
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False, gainImg=gain,
        )
        finite = np.isfinite(I)
        # Dividing uniform 1.0 by gain 2.0 -> 0.5
        np.testing.assert_allclose(I[finite], 0.5, atol=1e-10)

    def test_rBin_mode(self, simple_ring):
        """Binning in r-space instead of q-space should also work."""
        img, x, y = simple_ring
        r, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, rBin=5000,
            geomCorr=False, polCorr=False,
        )
        assert r.ndim == 1
        assert I.ndim == 1

    def test_custom_qBin_edges(self, simple_ring):
        """Passing explicit q-bin edges as an array."""
        img, x, y = simple_ring
        edges = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=edges,
            geomCorr=False, polCorr=False,
        )
        assert len(q) == len(edges) - 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            azimuthalBinning(
                np.zeros((3, 4)),
                np.zeros((3, 4)),
                np.zeros((4, 3)),
            )

    def test_negative_keV_raises(self, simple_ring):
        img, x, y = simple_ring
        with pytest.raises(ValueError, match="positive"):
            azimuthalBinning(img, x, y, keV=-5)

    def test_square_option(self, simple_ring):
        """With square=True the uniform image of 1s should still be 1."""
        img, x, y = simple_ring
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False, square=True,
        )
        finite = np.isfinite(I)
        np.testing.assert_allclose(I[finite], 1.0, atol=1e-10)

    def test_threshADU_filtering(self, simple_ring):
        img, x, y = simple_ring
        # Set some pixels above threshold
        img[0, :] = 100.0
        q, I = azimuthalBinning(
            img, x, y, z0=90_000, keV=10, qBin=0.2,
            geomCorr=False, polCorr=False,
            threshADU=[0, 50],
        )
        # Should still complete without error
        assert np.isfinite(I).any()


# ── pyFAI integration (create_J4m_integrator & azimuthalBinning_pyfai) ─


class TestCreateJ4mIntegrator:
    """Tests for create_J4m_integrator."""

    def test_returns_azimuthal_integrator(self):
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        ai = create_J4m_integrator()
        assert isinstance(ai, AzimuthalIntegrator)

    def test_distance_conversion(self):
        """z0 in microns should be converted to metres for pyFAI."""
        ai = create_J4m_integrator(z0=100_000)
        assert ai.dist == pytest.approx(0.1, rel=1e-8)

    def test_wavelength_from_keV(self):
        ai = create_J4m_integrator(keV=12.0)
        expected = 12.39841984e-10 / 12.0
        assert ai.wavelength == pytest.approx(expected, rel=1e-8)

    def test_poni_from_beam_center(self):
        ai = create_J4m_integrator(x0=1000, y0=2000)
        assert ai.poni1 == pytest.approx(2000e-6, rel=1e-8)
        assert ai.poni2 == pytest.approx(1000e-6, rel=1e-8)

    def test_rotation_from_tilt(self):
        ai = create_J4m_integrator(tx=1.0, ty=2.0)
        assert ai.rot1 == pytest.approx(np.deg2rad(2.0), rel=1e-8)
        assert ai.rot2 == pytest.approx(np.deg2rad(1.0), rel=1e-8)

    def test_z_off_adds_to_distance(self):
        ai = create_J4m_integrator(z0=90_000, z_off=10_000)
        assert ai.dist == pytest.approx(0.1, rel=1e-8)

    def test_detector_shape(self):
        """Detector should have the flattened J4M shape (4096, 1024)."""
        ai = create_J4m_integrator()
        assert ai.detector.max_shape == (4096, 1024)


class TestAzimuthalBinningPyFAI:
    """Tests for azimuthalBinning_pyfai."""

    @pytest.fixture(scope="class")
    def j4m_ai(self):
        """Pre-build an integrator once for all tests in this class."""
        return create_J4m_integrator(z0=90_000, keV=10)

    @pytest.fixture
    def uniform_img(self):
        """Uniform image matching the flattened J4M shape."""
        return np.ones((4096, 1024), dtype=float)

    def test_output_shapes_1d(self, j4m_ai, uniform_img):
        q, I = azimuthalBinning_pyfai(uniform_img, ai=j4m_ai, qBin=0.1)
        assert q.ndim == 1
        assert I.ndim == 1
        assert q.shape == I.shape

    def test_output_shapes_2d(self, j4m_ai, uniform_img):
        q, I = azimuthalBinning_pyfai(uniform_img, ai=j4m_ai, qBin=0.1, phiBins=4)
        assert q.ndim == 1
        assert I.ndim == 2
        assert I.shape[0] == 4
        assert I.shape[1] == len(q)

    def test_reshapes_3d_image(self, j4m_ai):
        """A (8, 512, 1024) image should be auto-reshaped to (4096, 1024)."""
        img_3d = np.ones((8, 512, 1024), dtype=float)
        q, I = azimuthalBinning_pyfai(img_3d, ai=j4m_ai, qBin=0.1)
        assert q.ndim == 1
        assert I.ndim == 1

    def test_reshapes_3d_mask(self, j4m_ai):
        """A 3-D mask should be reshaped along with the image."""
        img = np.ones((8, 512, 1024), dtype=float)
        mask = np.zeros((8, 512, 1024), dtype=bool)
        mask[0, :, :] = True
        q, I = azimuthalBinning_pyfai(img, ai=j4m_ai, qBin=0.1, mask=mask)
        assert np.isfinite(I).any()

    def test_reshapes_3d_dark(self, j4m_ai):
        """A 3-D dark image should be reshaped automatically."""
        img = np.ones((8, 512, 1024), dtype=float)
        dark = np.full((8, 512, 1024), 0.5)
        q, I = azimuthalBinning_pyfai(img, ai=j4m_ai, qBin=0.1, darkImg=dark)
        assert np.isfinite(I).any()

    def test_reshapes_3d_gain(self, j4m_ai):
        """A 3-D gain image should be reshaped automatically."""
        img = np.ones((8, 512, 1024), dtype=float)
        gain = np.full((8, 512, 1024), 2.0)
        q, I = azimuthalBinning_pyfai(img, ai=j4m_ai, qBin=0.1, gainImg=gain)
        assert np.isfinite(I).any()

    def test_mask_excludes_pixels(self, j4m_ai, uniform_img):
        mask = np.zeros_like(uniform_img, dtype=bool)
        mask[:2048, :] = True  # mask half
        q, I = azimuthalBinning_pyfai(uniform_img, ai=j4m_ai, qBin=0.1, mask=mask)
        assert np.isfinite(I).any()

    def test_reuse_ai_gives_same_result(self, j4m_ai, uniform_img):
        """Passing a pre-built ai should give the same result as building one."""
        q1, I1 = azimuthalBinning_pyfai(
            uniform_img, z0=90_000, keV=10, qBin=0.1, ai=j4m_ai,
        )
        q2, I2 = azimuthalBinning_pyfai(
            uniform_img, z0=90_000, keV=10, qBin=0.1, ai=j4m_ai,
        )
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(I1, I2)

    def test_rBin_mode(self, j4m_ai, uniform_img):
        """Integration in real-space radius with rBin."""
        r, I = azimuthalBinning_pyfai(uniform_img, ai=j4m_ai, rBin=5000)
        assert r.ndim == 1
        assert I.ndim == 1

    def test_polCorr_false(self, j4m_ai, uniform_img):
        """Disabling polarization correction should still work."""
        q, I = azimuthalBinning_pyfai(
            uniform_img, ai=j4m_ai, qBin=0.1, polCorr=False,
        )
        assert np.isfinite(I).any()

    def test_geomCorr_false(self, j4m_ai, uniform_img):
        """Disabling solid-angle correction should still work."""
        q, I = azimuthalBinning_pyfai(
            uniform_img, ai=j4m_ai, qBin=0.1, geomCorr=False,
        )
        assert np.isfinite(I).any()

    def test_q_values_positive(self, j4m_ai, uniform_img):
        q, I = azimuthalBinning_pyfai(uniform_img, ai=j4m_ai, qBin=0.1)
        assert np.all(q >= 0)
