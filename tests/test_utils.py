"""Tests for xrayscatteringtools.utils."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from xrayscatteringtools.utils import (
    au2invAngstroms,
    invAngstroms2au,
    keV2Angstroms,
    Angstroms2keV,
    q2theta,
    theta2q,
    compute_q_map,
    azimuthalBinning,
    element_symbol_to_number,
    element_number_to_symbol,
    translate_molecule,
    rotate_molecule,
    compress_ranges,
    ELEMENT_NUMBERS,
    ELEMENT_SYMBOLS,
    J4M,
)


# ── Unit conversion round-trips ────────────────────────────────────────


class TestUnitConversions:
    """Tests for simple unit-conversion helpers."""

    # -- au <-> invAngstroms --

    def test_au2invAngstroms_known_value(self):
        assert au2invAngstroms(1.0) == pytest.approx(1.8897261259077822)

    def test_invAngstroms2au_known_value(self):
        assert invAngstroms2au(1.8897261259077822) == pytest.approx(1.0)

    def test_au_invAngstroms_roundtrip(self):
        for val in [0.0, 0.5, 1.0, 3.7, 10.0]:
            assert invAngstroms2au(au2invAngstroms(val)) == pytest.approx(val)

    def test_invAngstroms_au_roundtrip(self):
        for val in [0.0, 0.5, 1.0, 5.0, 12.5]:
            assert au2invAngstroms(invAngstroms2au(val)) == pytest.approx(val)

    # -- keV <-> Angstroms --

    def test_keV2Angstroms_known_value(self):
        # 12.39841984 keV photon -> 1 Å
        assert keV2Angstroms(12.39841984) == pytest.approx(1.0)

    def test_Angstroms2keV_known_value(self):
        assert Angstroms2keV(1.0) == pytest.approx(12.39841984)

    def test_keV_Angstroms_roundtrip(self):
        for val in [1.0, 5.0, 10.0, 24.0]:
            assert Angstroms2keV(keV2Angstroms(val)) == pytest.approx(val)

    # -- q <-> theta --

    def test_q2theta_zero(self):
        assert q2theta(0.0, 10.0) == pytest.approx(0.0)

    def test_theta2q_zero(self):
        assert theta2q(0.0, 10.0) == pytest.approx(0.0)

    def test_q_theta_roundtrip(self):
        keV = 10.0
        for q in [0.5, 1.0, 3.0, 5.0]:
            assert theta2q(q2theta(q, keV), keV) == pytest.approx(q)

    def test_theta_q_roundtrip(self):
        keV = 10.0
        for theta in [0.01, 0.1, 0.5, 1.0]:
            assert q2theta(theta2q(theta, keV), keV) == pytest.approx(theta)

    def test_q2theta_array(self):
        q = np.array([0.0, 1.0, 2.0])
        result = q2theta(q, 10.0)
        assert result.shape == q.shape
        assert result[0] == pytest.approx(0.0)

    def test_theta2q_array(self):
        theta = np.array([0.0, 0.1, 0.5])
        result = theta2q(theta, 10.0)
        assert result.shape == theta.shape
        assert result[0] == pytest.approx(0.0)


# ── Element lookups ────────────────────────────────────────────────────


class TestElementLookups:
    """Tests for element symbol/number conversion."""

    def test_symbol_to_number_common(self):
        assert element_symbol_to_number("H") == 1
        assert element_symbol_to_number("C") == 6
        assert element_symbol_to_number("O") == 8
        assert element_symbol_to_number("Fe") == 26
        assert element_symbol_to_number("Og") == 118

    def test_number_to_symbol_common(self):
        assert element_number_to_symbol(1) == "H"
        assert element_number_to_symbol(6) == "C"
        assert element_number_to_symbol(8) == "O"
        assert element_number_to_symbol(26) == "Fe"
        assert element_number_to_symbol(118) == "Og"

    def test_symbol_to_number_invalid(self):
        with pytest.raises(KeyError):
            element_symbol_to_number("Xx")

    def test_number_to_symbol_invalid(self):
        with pytest.raises(KeyError):
            element_number_to_symbol(0)
        with pytest.raises(KeyError):
            element_number_to_symbol(999)

    def test_roundtrip_all_elements(self):
        for symbol, number in ELEMENT_NUMBERS.items():
            assert element_number_to_symbol(number) == symbol
            assert element_symbol_to_number(symbol) == number

    def test_dicts_consistent(self):
        assert len(ELEMENT_NUMBERS) == len(ELEMENT_SYMBOLS)
        assert len(ELEMENT_NUMBERS) == 118
        for sym, num in ELEMENT_NUMBERS.items():
            assert ELEMENT_SYMBOLS[num] == sym


# ── Molecular geometry helpers ─────────────────────────────────────────


class TestTranslateMolecule:
    def test_translate_identity(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = translate_molecule(coords, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(result, coords)

    def test_translate_simple(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        result = translate_molecule(coords, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0, 3.0]])

    def test_translate_preserves_shape(self):
        coords = np.random.rand(5, 3)
        vec = np.array([1.0, -1.0, 0.5])
        result = translate_molecule(coords, vec)
        assert result.shape == coords.shape


class TestRotateMolecule:
    def test_identity_rotation(self):
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = rotate_molecule(coords, 0, 0, 0)
        np.testing.assert_array_almost_equal(result, coords)

    def test_rotate_90_about_z(self):
        # 90° about z: (1,0,0) -> (0,1,0)
        coords = np.array([[1.0, 0.0, 0.0]])
        result = rotate_molecule(coords, alpha=0, beta=0, gamma=90)
        np.testing.assert_array_almost_equal(result, [[0.0, 1.0, 0.0]], decimal=10)

    def test_rotate_90_about_y(self):
        # 90° about y: (1,0,0) -> (0,0,-1)
        coords = np.array([[1.0, 0.0, 0.0]])
        result = rotate_molecule(coords, alpha=0, beta=90, gamma=0)
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, -1.0]], decimal=10)

    def test_rotate_90_about_x(self):
        # 90° about x: (0,1,0) -> (0,0,1)
        coords = np.array([[0.0, 1.0, 0.0]])
        result = rotate_molecule(coords, alpha=90, beta=0, gamma=0)
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, 1.0]], decimal=10)

    def test_360_rotation_roundtrip(self):
        coords = np.random.rand(4, 3)
        result = rotate_molecule(coords, 360, 360, 360)
        np.testing.assert_array_almost_equal(result, coords)

    def test_rotation_preserves_distances(self):
        """Rotation should preserve pairwise distances."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = rotate_molecule(coords, 30, 45, 60)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d_orig = np.linalg.norm(coords[i] - coords[j])
                d_rot = np.linalg.norm(rotated[i] - rotated[j])
                assert d_rot == pytest.approx(d_orig, abs=1e-12)


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


# ── compress_ranges ────────────────────────────────────────────────────


class TestCompressRanges:
    def test_single_value(self):
        assert compress_ranges([5]) == "5"

    def test_consecutive(self):
        assert compress_ranges([1, 2, 3]) == "1-3"

    def test_non_consecutive(self):
        assert compress_ranges([1, 3, 5]) == "1,3,5"

    def test_mixed(self):
        assert compress_ranges([1, 2, 3, 5, 7, 8, 9]) == "1-3,5,7-9"

    def test_unsorted_input(self):
        assert compress_ranges([9, 1, 3, 2, 8, 7, 5]) == "1-3,5,7-9"

    def test_duplicates(self):
        assert compress_ranges([1, 1, 2, 2, 3]) == "1-3"

    def test_negative_numbers(self):
        assert compress_ranges([-3, -2, -1, 0, 1]) == "-3-1"

    def test_single_pair(self):
        assert compress_ranges([10, 11]) == "10-11"

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            compress_ranges([])


# ── J4M lazy loader ────────────────────────────────────────────────────


class TestJ4M:
    def test_has_expected_attributes(self):
        assert hasattr(J4M, "x")
        assert hasattr(J4M, "y")

    def test_x_y_shapes_match(self):
        assert J4M.x.shape == J4M.y.shape

    def test_repr_after_load(self):
        # Accessing an attribute forces loading
        _ = J4M.x
        r = repr(J4M)
        assert "not yet loaded" not in r
