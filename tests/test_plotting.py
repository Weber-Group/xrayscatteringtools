"""Tests for xrayscatteringtools.plotting."""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from xrayscatteringtools.plotting import (
    compute_pixel_edges,
    edges_from_centers,
    plot_jungfrau,
    plot_j4m,
)

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")


# ── compute_pixel_edges ────────────────────────────────────────────────


class TestComputePixelEdges:
    """Tests for the 2-D pixel-edge computation."""

    def test_output_shape(self):
        coord = np.zeros((4, 5))
        edges = compute_pixel_edges(coord)
        assert edges.shape == (5, 6)

    def test_output_shape_square(self):
        coord = np.zeros((10, 10))
        edges = compute_pixel_edges(coord)
        assert edges.shape == (11, 11)

    def test_uniform_grid(self):
        """For a regular grid the edges should land at half-pixel offsets."""
        rows, cols = 3, 4
        y, x = np.mgrid[0:rows, 0:cols]  # centers at integer positions
        x = x.astype(float)
        edges = compute_pixel_edges(x)

        # Interior edges should be at 0.5-offset midpoints
        for i in range(1, rows):
            for j in range(1, cols):
                expected = 0.25 * (x[i - 1, j - 1] + x[i, j - 1] +
                                   x[i - 1, j] + x[i, j])
                assert edges[i, j] == pytest.approx(expected)

    def test_corners_extrapolated(self):
        """Corners should extrapolate symmetrically for a regular grid."""
        coord = np.arange(12, dtype=float).reshape(3, 4)
        edges = compute_pixel_edges(coord)
        # Top-left corner
        expected_tl = (
            coord[0, 0]
            - 0.5 * (coord[1, 0] - coord[0, 0])
            - 0.5 * (coord[0, 1] - coord[0, 0])
        )
        assert edges[0, 0] == pytest.approx(expected_tl)
        # Bottom-right corner
        expected_br = (
            coord[-1, -1]
            + 0.5 * (coord[-1, -1] - coord[-2, -1])
            + 0.5 * (coord[-1, -1] - coord[-1, -2])
        )
        assert edges[-1, -1] == pytest.approx(expected_br)

    def test_monotonic_edges(self):
        """Edges computed from a monotonically increasing grid should also
        be monotonically increasing along each axis."""
        rows, cols = 5, 6
        y, x = np.mgrid[0:rows, 0:cols]
        x = x.astype(float)
        edges = compute_pixel_edges(x)
        # Each row should be monotonically increasing
        for i in range(edges.shape[0]):
            assert np.all(np.diff(edges[i, :]) > -1e-12)

    def test_minimum_size(self):
        """Smallest valid input: 2x2 centers -> 3x3 edges."""
        coord = np.array([[0.0, 1.0], [0.0, 1.0]])
        edges = compute_pixel_edges(coord)
        assert edges.shape == (3, 3)


# ── edges_from_centers ─────────────────────────────────────────────────


class TestEdgesFromCenters:
    """Tests for 1-D bin-edge computation from centers."""

    def test_output_length(self):
        centers = np.array([1.0, 2.0, 3.0, 4.0])
        edges = edges_from_centers(centers)
        assert len(edges) == len(centers) + 1

    def test_uniform_spacing(self):
        centers = np.array([1.0, 2.0, 3.0])
        edges = edges_from_centers(centers)
        np.testing.assert_allclose(edges, [0.5, 1.5, 2.5, 3.5])

    def test_non_uniform_spacing(self):
        centers = np.array([0.0, 1.0, 3.0])
        edges = edges_from_centers(centers)
        # first edge: 0 - (1-0)/2 = -0.5
        assert edges[0] == pytest.approx(-0.5)
        # inner edge: (0+1)/2 = 0.5
        assert edges[1] == pytest.approx(0.5)
        # inner edge: (1+3)/2 = 2.0
        assert edges[2] == pytest.approx(2.0)
        # last edge: 3 + (3-1)/2 = 4.0
        assert edges[3] == pytest.approx(4.0)

    def test_two_centers(self):
        centers = np.array([5.0, 10.0])
        edges = edges_from_centers(centers)
        np.testing.assert_allclose(edges, [2.5, 7.5, 12.5])

    def test_single_center_raises(self):
        with pytest.raises(ValueError, match="at least 2 elements"):
            edges_from_centers(np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 2 elements"):
            edges_from_centers(np.array([]))

    def test_list_input(self):
        """Should accept plain Python lists via np.asarray."""
        edges = edges_from_centers([1, 2, 3])
        assert len(edges) == 4

    def test_edges_bracket_centers(self):
        """Each center should lie between its two surrounding edges."""
        centers = np.array([0.0, 0.3, 1.5, 4.0, 4.1])
        edges = edges_from_centers(centers)
        for i, c in enumerate(centers):
            assert edges[i] < c < edges[i + 1]

    def test_negative_centers(self):
        centers = np.array([-3.0, -2.0, -1.0])
        edges = edges_from_centers(centers)
        np.testing.assert_allclose(edges, [-3.5, -2.5, -1.5, -0.5])


# ── plot_jungfrau ──────────────────────────────────────────────────────


class TestPlotJungfrau:
    """Tests for the Jungfrau tile-plotting function."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")

    @staticmethod
    def _make_tiles(n_tiles=8, rows=4, cols=6):
        """Create small synthetic tile data for testing."""
        x = [np.random.rand(rows, cols) * 100 for _ in range(n_tiles)]
        y = [np.random.rand(rows, cols) * 100 for _ in range(n_tiles)]
        f = [np.random.rand(rows, cols) for _ in range(n_tiles)]
        return x, y, f

    def test_returns_quadmesh(self):
        x, y, f = self._make_tiles()
        pcm = plot_jungfrau(x, y, f)
        assert isinstance(pcm, matplotlib.collections.QuadMesh)

    def test_uses_provided_axes(self):
        x, y, f = self._make_tiles()
        fig, ax = plt.subplots()
        pcm = plot_jungfrau(x, y, f, ax=ax)
        # The QuadMesh should be attached to the provided axes
        assert pcm.axes is ax

    def test_creates_axes_when_none(self):
        x, y, f = self._make_tiles()
        pcm = plot_jungfrau(x, y, f, ax=None)
        assert pcm.axes is not None

    def test_aspect_equal(self):
        x, y, f = self._make_tiles()
        fig, ax = plt.subplots()
        plot_jungfrau(x, y, f, ax=ax)
        # get_aspect() may return the string "equal" or the numeric 1.0
        assert ax.get_aspect() in ("equal", 1.0)

    def test_explicit_vmin_vmax(self):
        x, y, f = self._make_tiles()
        pcm = plot_jungfrau(x, y, f, vmin=-1, vmax=5)
        assert pcm.get_clim() == (-1, 5)

    def test_auto_vmin_vmax(self):
        x, y, f = self._make_tiles()
        pcm = plot_jungfrau(x, y, f)
        clim = pcm.get_clim()
        all_vals = np.concatenate([fi.ravel() for fi in f])
        assert clim[0] == pytest.approx(np.nanmin(all_vals))
        assert clim[1] == pytest.approx(np.nanmax(all_vals))


# ── plot_j4m ───────────────────────────────────────────────────────────


class TestPlotJ4M:
    """Tests for the J4M convenience wrapper."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")

    def test_with_explicit_coords(self):
        """When x/y are provided, should use them directly."""
        rows, cols = 4, 6
        x = np.array([np.random.rand(rows, cols) for _ in range(8)])
        y = np.array([np.random.rand(rows, cols) for _ in range(8)])
        f = np.array([np.random.rand(rows, cols) for _ in range(8)])
        pcm = plot_j4m(f, x=x, y=y)
        assert isinstance(pcm, matplotlib.collections.QuadMesh)

    @patch("xrayscatteringtools.plotting.J4M")
    def test_uses_j4m_coords_when_none(self, mock_j4m):
        """When x/y are None, coordinates should come from J4M."""
        rows, cols = 4, 6
        mock_j4m.x = np.array([np.arange(rows * cols, dtype=float).reshape(rows, cols)
                                for _ in range(8)])
        mock_j4m.y = np.array([np.arange(rows * cols, dtype=float).reshape(rows, cols)
                                for _ in range(8)])
        f = np.array([np.random.rand(rows, cols) for _ in range(8)])
        pcm = plot_j4m(f)
        assert isinstance(pcm, matplotlib.collections.QuadMesh)
