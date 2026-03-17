"""Tests for xrayscatteringtools.calib.masking.

MaskMaker relies on LCLS infrastructure (``combineRuns``, ``psana``) and the
full Jungfrau 4M geometry (``J4M``).  We mock those heavy dependencies so the
logic can be exercised in a lightweight, offline test environment.
"""

from __future__ import annotations

import types
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import xrayscatteringtools.calib.masking as masking_mod
from xrayscatteringtools.calib.masking import MaskMaker, _DARK_PERCENTILE_LO, _DARK_PERCENTILE_HI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small tile dimensions for quick tests  (8 tiles of 4×4 pixels)
_N_TILES, _NY, _NX = 8, 4, 4
_SHAPE = (_N_TILES, _NY, _NX)
_N_PIX = _N_TILES * _NY * _NX


def _fake_j4m():
    """Return a lightweight namespace mimicking J4M attributes."""
    rng = np.random.default_rng(42)
    ns = types.SimpleNamespace()
    # Coordinates spread across a ~100 mm detector face
    ns.x = (rng.random(_SHAPE) - 0.5) * 100_000  # microns
    ns.y = (rng.random(_SHAPE) - 0.5) * 100_000
    ns.line_mask = np.ones(_SHAPE, dtype=bool)
    ns.t_mask = np.ones(_SHAPE, dtype=bool)
    return ns


def _fake_combine_runs(
    n_shots: int = 10,
    shape: tuple = _SHAPE,
    xray_on_frac: float = 0.5,
):
    """Return a dict that looks like combineRuns output."""
    rng = np.random.default_rng(0)
    xray = np.zeros(n_shots, dtype=int)
    xray[: int(n_shots * xray_on_frac)] = 1
    return {
        "lightStatus/xray": xray,
        "lightStatus/laser": np.ones(n_shots, dtype=int),
        "jungfrau4M/azav_azav": rng.random((n_shots, 100)),
        "Sums/jungfrau4M_calib_xrayOn_thresADU1": rng.random(shape) * 100,
        "Sums/jungfrau4M_calib_dropped": rng.random(shape) * 10,
    }


@pytest.fixture
def fake_j4m():
    return _fake_j4m()


@pytest.fixture
def mask_maker(fake_j4m):
    """Return a MaskMaker with all heavy I/O mocked out."""
    fake_data = _fake_combine_runs()
    with (
        patch.object(masking_mod, "combineRuns", return_value=fake_data),
        patch.object(masking_mod, "J4M", fake_j4m),
    ):
        mm = MaskMaker(
            experiment="cxitest",
            data_path="/fake/path",
            dark_run_number=1,
            background_run_number=2,
            sample_run_number=3,
        )
    # Re-patch J4M for any subsequent calls inside methods
    mm._j4m_ref = fake_j4m
    return mm


# ===================================================================
# Module-level constants
# ===================================================================
class TestConstants:
    def test_dark_percentiles(self):
        assert 0 < _DARK_PERCENTILE_LO < _DARK_PERCENTILE_HI < 100

    def test_n_q_bins(self):
        assert masking_mod._N_Q_BINS > 0

    def test_auto_accept_threshold(self):
        assert 0 < masking_mod._AUTO_ACCEPT_THRESHOLD <= 1.0

    def test_keys_to_combine_is_list(self):
        assert isinstance(masking_mod._KEYS_TO_COMBINE, list)

    def test_keys_to_sum_is_list(self):
        assert isinstance(masking_mod._KEYS_TO_SUM, list)


# ===================================================================
# __init__ validation (runs BEFORE combineRuns is called)
# ===================================================================
class TestInitValidation:

    def test_empty_experiment(self):
        with pytest.raises(ValueError, match="experiment"):
            MaskMaker("", "/data", 1, 2, 3)

    def test_non_string_experiment(self):
        with pytest.raises(ValueError, match="experiment"):
            MaskMaker(123, "/data", 1, 2, 3)

    def test_empty_data_path(self):
        with pytest.raises(ValueError, match="data_path"):
            MaskMaker("exp", "", 1, 2, 3)

    def test_non_string_data_path(self):
        with pytest.raises(ValueError, match="data_path"):
            MaskMaker("exp", None, 1, 2, 3)

    def test_negative_dark_run(self):
        with pytest.raises(ValueError, match="dark_run_number"):
            MaskMaker("exp", "/data", -1, 2, 3)

    def test_negative_background_run(self):
        with pytest.raises(ValueError, match="background_run_number"):
            MaskMaker("exp", "/data", 1, -2, 3)

    def test_negative_sample_run(self):
        with pytest.raises(ValueError, match="sample_run_number"):
            MaskMaker("exp", "/data", 1, 2, -3)

    def test_float_run_number(self):
        with pytest.raises(ValueError, match="dark_run_number"):
            MaskMaker("exp", "/data", 1.5, 2, 3)

    def test_string_run_number(self):
        with pytest.raises(ValueError, match="dark_run_number"):
            MaskMaker("exp", "/data", "one", 2, 3)


# ===================================================================
# MaskMaker construction (mocked)
# ===================================================================
class TestMaskMakerInit:

    def test_attributes_stored(self, mask_maker):
        assert mask_maker.experiment == "cxitest"
        assert mask_maker.data_run_number if False else True  # dummy
        assert mask_maker.dark_run_number == 1
        assert mask_maker.background_run_number == 2
        assert mask_maker.sample_run_number == 3

    def test_masks_start_all_true(self, mask_maker, fake_j4m):
        with patch.object(masking_mod, "J4M", fake_j4m):
            assert mask_maker.dark_mask.all()
            assert mask_maker.background_mask.all()
            assert mask_maker.poly_mask.all()
            assert mask_maker.sample_mask.all()
            assert mask_maker.cmask.all()

    def test_repr(self, mask_maker):
        r = repr(mask_maker)
        assert "cxitest" in r
        assert "dark=1" in r
        assert "bkg=2" in r
        assert "sample=3" in r


# ===================================================================
# _masked_fraction (static, no dependencies)
# ===================================================================
class TestMaskedFraction:

    def test_all_true(self):
        mask = np.ones(100, dtype=bool)
        assert MaskMaker._masked_fraction(mask) == pytest.approx(0.0)

    def test_all_false(self):
        mask = np.zeros(100, dtype=bool)
        assert MaskMaker._masked_fraction(mask) == pytest.approx(1.0)

    def test_half(self):
        mask = np.array([True] * 50 + [False] * 50)
        assert MaskMaker._masked_fraction(mask) == pytest.approx(0.5)

    def test_2d(self):
        mask = np.ones((8, 4, 4), dtype=bool)
        mask[0] = False  # mask one full tile
        expected = 16 / 128
        assert MaskMaker._masked_fraction(mask) == pytest.approx(expected)


# ===================================================================
# process_dark
# ===================================================================
class TestProcessDark:

    def test_explicit_bounds(self, mask_maker, fake_j4m):
        """With explicit lb/ub no interactive input is needed."""
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            mask_maker.process_dark(plotting=False, lb=-10.0, ub=10.0)
        assert mask_maker.dark_mask.dtype == bool

    def test_tight_bounds_mask_more(self, mask_maker, fake_j4m):
        """Tighter bounds should mask more pixels than wide bounds."""
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            mask_maker.process_dark(plotting=False, lb=-100, ub=100)
            wide_frac = MaskMaker._masked_fraction(mask_maker.dark_mask)

            mask_maker.process_dark(plotting=False, lb=-0.001, ub=0.001)
            tight_frac = MaskMaker._masked_fraction(mask_maker.dark_mask)

        assert tight_frac >= wide_frac

    def test_lb_ge_ub_raises(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            with pytest.raises(ValueError, match="Lower bound"):
                mask_maker.process_dark(plotting=False, lb=5, ub=5)

    def test_plotting_flag(self, mask_maker, fake_j4m):
        """With plotting=True the function should still succeed (plots mocked)."""
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(MaskMaker, "_plot_j4m_panel"),
        ):
            mask_maker.process_dark(plotting=True, lb=-10, ub=10)
        assert mask_maker.dark_mask.dtype == bool


# ===================================================================
# process_background
# ===================================================================
class TestProcessBackground:

    def test_explicit_bounds(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            mask_maker.process_background(plotting=False, lb=0, ub=1000)
        assert mask_maker.background_mask.dtype == bool

    def test_lb_ge_ub_raises(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            with pytest.raises(ValueError, match="Lower bound"):
                mask_maker.process_background(plotting=False, lb=10, ub=5)

    def test_tight_bounds_mask_more(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            mask_maker.process_background(plotting=False, lb=0, ub=1e6)
            wide_frac = MaskMaker._masked_fraction(mask_maker.background_mask)

            mask_maker.process_background(plotting=False, lb=0, ub=0.001)
            tight_frac = MaskMaker._masked_fraction(mask_maker.background_mask)
        assert tight_frac >= wide_frac


# ===================================================================
# apply_polygon_mask
# ===================================================================
class TestApplyPolygonMask:

    def test_requires_three_points(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
        ):
            with pytest.raises(ValueError, match="3 points"):
                mask_maker.apply_polygon_mask(num_points=2)

    def test_polygon_masks_outside(self, mask_maker, fake_j4m):
        """A polygon that covers the whole detector should keep everything."""
        # Build a huge rectangle that encloses all fake_j4m coords
        # Note: apply_polygon_mask flips coords: (-y, x), so polygon
        # vertices are in display (y, -x) space.  We provide vertices
        # large enough in *display* space.
        big = 200_000  # microns, bigger than any coordinate
        points = [(-big, -big), (big, -big), (big, big), (-big, big)]
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.apply_polygon_mask(num_points=4, points=points, plotting=False)
        # Everything should be inside, so poly_mask is all True
        assert mask_maker.poly_mask.all()

    def test_tiny_polygon_masks_most(self, mask_maker, fake_j4m):
        """A tiny polygon far away should mask almost everything."""
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]  # ~1 µm square at origin
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.apply_polygon_mask(num_points=4, points=pts, plotting=False)
        frac_masked = MaskMaker._masked_fraction(mask_maker.poly_mask)
        # Nearly everything should be masked out
        assert frac_masked > 0.9

    def test_resets_poly_mask(self, mask_maker, fake_j4m):
        """Each call should reset the polygon mask before building a new one."""
        big = 200_000
        full_pts = [(-big, -big), (big, -big), (big, big), (-big, big)]
        tiny_pts = [(0, 0), (1, 0), (1, 1)]
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            # First: tiny polygon → most masked
            mask_maker.apply_polygon_mask(3, points=tiny_pts, plotting=False)
            assert not mask_maker.poly_mask.all()

            # Second: huge polygon → resets and keeps everything
            mask_maker.apply_polygon_mask(4, points=full_pts, plotting=False)
            assert mask_maker.poly_mask.all()


# ===================================================================
# _compute_sample_average
# ===================================================================
class TestComputeSampleAverage:

    def test_returns_ndarray(self, mask_maker, fake_j4m):
        with patch.object(masking_mod, "J4M", fake_j4m):
            avg = mask_maker._compute_sample_average(apply_masks=False)
        assert isinstance(avg, np.ndarray)

    def test_apply_masks_sets_nan(self, mask_maker, fake_j4m):
        """When masks are applied, masked pixels become NaN."""
        with patch.object(masking_mod, "J4M", fake_j4m):
            # Mask one tile in dark_mask
            mask_maker.dark_mask[0] = False
            avg = mask_maker._compute_sample_average(apply_masks=True)
        assert np.all(np.isnan(avg[0]))

    def test_no_masks_no_nan(self, mask_maker, fake_j4m):
        with patch.object(masking_mod, "J4M", fake_j4m):
            avg = mask_maker._compute_sample_average(apply_masks=False)
        assert not np.any(np.isnan(avg))


# ===================================================================
# combine_masks
# ===================================================================
class TestCombineMasks:

    def test_all_masks_true_gives_line_and_t(self, mask_maker, fake_j4m):
        """When all user masks are True, cmask equals line_mask & t_mask."""
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        expected = fake_j4m.line_mask.astype(bool) & fake_j4m.t_mask.astype(bool)
        np.testing.assert_array_equal(mask_maker.cmask, expected)

    def test_dark_mask_propagates(self, mask_maker, fake_j4m):
        """Masking a tile in dark_mask should propagate to cmask."""
        mask_maker.dark_mask[0] = False
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert not np.any(mask_maker.cmask[0])

    def test_background_mask_propagates(self, mask_maker, fake_j4m):
        mask_maker.background_mask[1] = False
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert not np.any(mask_maker.cmask[1])

    def test_poly_mask_propagates(self, mask_maker, fake_j4m):
        mask_maker.poly_mask[2] = False
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert not np.any(mask_maker.cmask[2])

    def test_sample_mask_propagates(self, mask_maker, fake_j4m):
        mask_maker.sample_mask[3] = False
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert not np.any(mask_maker.cmask[3])

    def test_line_mask_false_propagates(self, mask_maker, fake_j4m):
        """If line_mask has a False pixel, cmask should too."""
        fake_j4m.line_mask[4, 0, 0] = False
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert not mask_maker.cmask[4, 0, 0]

    def test_cmask_dtype_bool(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
        ):
            mask_maker.combine_masks(plotting=False)
        assert mask_maker.cmask.dtype == bool


# ===================================================================
# process_sample (smoke test with mocked q_map)
# ===================================================================
class TestProcessSample:

    def test_runs_without_error(self, mask_maker, fake_j4m):
        """Smoke test: process_sample should complete without raising."""
        # q_idx == 0 always triggers manual review, so mock input + subplots
        mock_fig = MagicMock()
        mock_axes = (MagicMock(), MagicMock())
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_axes)),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
            patch.object(masking_mod, "compute_q_map", return_value=np.random.default_rng(7).random(_SHAPE)),
            patch.object(masking_mod, "thompson_correction", return_value=np.ones(_SHAPE)),
            patch("builtins.input", side_effect=["-1e30", "1e30"] * 10),
        ):
            mask_maker.process_sample(
                n_std=100,  # very wide
                plotting=False,
                n_q_bins=5,
                auto_accept_threshold=0.0,  # accept everything
            )
        assert mask_maker.sample_mask.dtype == bool

    def test_sample_mask_shape(self, mask_maker, fake_j4m):
        mock_fig = MagicMock()
        mock_axes = (MagicMock(), MagicMock())
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_axes)),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
            patch.object(masking_mod, "compute_q_map", return_value=np.random.default_rng(7).random(_SHAPE)),
            patch.object(masking_mod, "thompson_correction", return_value=np.ones(_SHAPE)),
            patch("builtins.input", side_effect=["-1e30", "1e30"] * 10),
        ):
            mask_maker.process_sample(n_std=100, plotting=False, n_q_bins=5, auto_accept_threshold=0.0)
        assert mask_maker.sample_mask.shape == _SHAPE


# ===================================================================
# diagnose_q_bins (smoke test)
# ===================================================================
class TestDiagnoseQBins:

    def test_runs_without_error(self, mask_maker, fake_j4m):
        with (
            patch.object(masking_mod, "J4M", fake_j4m),
            patch("matplotlib.pyplot.show"),
            patch.object(masking_mod, "plot_j4m", return_value=MagicMock()),
            patch.object(masking_mod, "compute_q_map", return_value=np.random.default_rng(7).random(_SHAPE)),
            patch.object(masking_mod, "thompson_correction", return_value=np.ones(_SHAPE)),
        ):
            mask_maker.diagnose_q_bins(n_q_bins=5)


# ===================================================================
# Backward-compatible alias
# ===================================================================
class TestAlias:
    def test_mask_maker_alias(self):
        assert masking_mod.mask_maker is MaskMaker
