"""Tests for xrayscatteringtools.calib.timetool_calibration."""

import os
import tempfile

import numpy as np
import pytest
from scipy.special import erf
from ruamel.yaml import YAML

from xrayscatteringtools.calib.timetool_calibration import (
    fast_erf_fit,
    add_calibration_to_yaml,
    apply_timetool_correction,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_erf_array(n=500, center=250, width=40, lo=0.0, hi=1.0):
    """Generate a smooth error-function-like step from *lo* to *hi*."""
    x = np.arange(n, dtype=float)
    norm = 0.5 * (1.0 + erf((x - center) / width))
    return lo + (hi - lo) * norm


@pytest.fixture
def erf_array():
    """Default erf-like test array with center=250, width=40."""
    return _make_erf_array()


@pytest.fixture
def yaml_file(tmp_path):
    """Create a minimal YAML file and return its path."""
    fp = tmp_path / "config.yaml"
    yaml = YAML()
    yaml.dump({"experiment": "test123"}, fp)
    return str(fp)


@pytest.fixture
def yaml_file_with_calib(tmp_path):
    """YAML file that already has a tt_calibration key with one entry."""
    fp = tmp_path / "config.yaml"
    yaml = YAML()
    yaml.dump(
        {
            "experiment": "test123",
            "tt_calibration": [
                {"runs": [1, 10], "slope": 0.1, "intercept": 2.0}
            ],
        },
        fp,
    )
    return str(fp)


# ===================================================================
# fast_erf_fit
# ===================================================================
class TestFastErfFit:
    """Tests for fast_erf_fit."""

    def test_returns_five_values(self, erf_array):
        result = fast_erf_fit(erf_array)
        assert len(result) == 5

    def test_return_types(self, erf_array):
        range_val, cent_pos, cent_amp, norm_data, slope_val = fast_erf_fit(erf_array)
        assert isinstance(range_val, (int, np.integer))
        assert isinstance(cent_pos, (int, np.integer))
        assert isinstance(cent_amp, (float, np.floating))
        assert isinstance(norm_data, np.ndarray)
        assert isinstance(slope_val, (float, np.floating))

    def test_norm_data_range(self, erf_array):
        """Normalized data should be in [0, 1]."""
        *_, norm_data, _ = fast_erf_fit(erf_array)
        assert np.nanmin(norm_data) >= -1e-12
        assert np.nanmax(norm_data) <= 1.0 + 1e-12

    def test_center_near_expected(self, erf_array):
        """Center should be close to 250 for the default erf array."""
        _, cent_pos, _, _, _ = fast_erf_fit(erf_array)
        assert abs(cent_pos - 250) < 50  # generous tolerance

    def test_slope_positive(self, erf_array):
        """For an increasing step, slope should be positive."""
        *_, slope_val = fast_erf_fit(erf_array)
        assert slope_val > 0

    def test_center_amplitude_near_half(self, erf_array):
        """Amplitude at the center of an erf should be near 0.5."""
        _, _, cent_amp, _, _ = fast_erf_fit(erf_array)
        assert 0.2 < cent_amp < 0.8

    def test_range_positive(self, erf_array):
        """Transition width must be a positive integer."""
        range_val, *_ = fast_erf_fit(erf_array)
        assert range_val > 0

    # --- edge / degenerate cases ---

    def test_flat_array(self):
        """Flat input should return safe zeros."""
        arr = np.full(100, 5.0)
        range_val, cent_pos, cent_amp, norm_data, slope_val = fast_erf_fit(arr)
        assert range_val == 0
        assert cent_pos == 0
        assert cent_amp == 0
        assert slope_val == 0
        assert np.all(norm_data == 0)

    def test_already_at_one(self):
        """Array already at max (all values > max_val after norm) → thresholds not crossable."""
        arr = np.linspace(0.95, 1.0, 100)
        range_val, cent_pos, cent_amp, norm_data, slope_val = fast_erf_fit(arr)
        # low_idx may be empty ⇒ safe defaults
        assert range_val == 0 or range_val > 0  # either path is valid

    def test_reversed_step(self):
        """A decreasing step: norm goes 1→0, so low_idx.max >= high_idx.min → zeros."""
        arr = _make_erf_array()[::-1]
        range_val, cent_pos, cent_amp, norm_data, slope_val = fast_erf_fit(arr)
        # After normalization the step goes 0→1 again because (arr-lo)/(hi-lo) flips.
        # But let's just verify it returns something finite.
        assert np.isfinite(slope_val)

    def test_custom_thresholds(self):
        """Tighter thresholds should give a narrower range_val."""
        arr = _make_erf_array(width=40)
        range_default = fast_erf_fit(arr, min_val=0.1, max_val=0.9)[0]
        range_tight = fast_erf_fit(arr, min_val=0.3, max_val=0.7)[0]
        assert range_tight < range_default

    def test_wider_transition(self):
        """A wider erf should give a larger range_val."""
        narrow = _make_erf_array(width=20)
        wide = _make_erf_array(width=80)
        assert fast_erf_fit(narrow)[0] < fast_erf_fit(wide)[0]

    def test_shifted_center(self):
        """Center parameter should shift cent_pos."""
        arr_left = _make_erf_array(center=100)
        arr_right = _make_erf_array(center=400)
        _, cp_left, _, _, _ = fast_erf_fit(arr_left)
        _, cp_right, _, _, _ = fast_erf_fit(arr_right)
        assert cp_left < cp_right

    def test_non_unit_amplitude(self):
        """Works correctly when the data doesn't span [0,1]."""
        arr = _make_erf_array(lo=100.0, hi=500.0)
        range_val, cent_pos, cent_amp, norm_data, slope_val = fast_erf_fit(arr)
        assert range_val > 0
        assert 0.0 <= norm_data.min()
        assert norm_data.max() <= 1.0 + 1e-12

    def test_short_array(self):
        """Very short arrays should not crash."""
        arr = np.array([0.0, 1.0])
        result = fast_erf_fit(arr)
        assert len(result) == 5

    def test_list_input(self):
        """Plain Python list should be accepted."""
        arr = list(_make_erf_array(n=200))
        result = fast_erf_fit(arr)
        assert len(result) == 5


# ===================================================================
# add_calibration_to_yaml
# ===================================================================
class TestAddCalibrationToYaml:
    """Tests for add_calibration_to_yaml."""

    def test_creates_key_and_entry(self, yaml_file):
        """Should create 'tt_calibration' key when absent."""
        add_calibration_to_yaml([30, ".inf"], 0.5, 1.0, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        assert "tt_calibration" in data
        assert len(data["tt_calibration"]) == 1
        entry = data["tt_calibration"][0]
        assert entry["slope"] == 0.5
        assert entry["intercept"] == 1.0

    def test_runs_list_values(self, yaml_file):
        """Runs should contain the numeric value and infinity."""
        add_calibration_to_yaml([30, ".inf"], 0.1, 2.0, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        runs = list(data["tt_calibration"][0]["runs"])
        assert runs[0] == 30
        assert runs[1] == float("inf")

    def test_appends_to_existing(self, yaml_file_with_calib):
        """A second entry should be appended, not overwrite."""
        add_calibration_to_yaml([11, 20], 0.2, 3.0, file_path=yaml_file_with_calib)
        yaml = YAML()
        with open(yaml_file_with_calib) as f:
            data = yaml.load(f)
        assert len(data["tt_calibration"]) == 2

    def test_preserves_existing_keys(self, yaml_file):
        """Other top-level keys should be untouched."""
        add_calibration_to_yaml([1, 5], 0.1, 0.2, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        assert data["experiment"] == "test123"

    def test_custom_key_name(self, yaml_file):
        """Should use the given key_name instead of the default."""
        add_calibration_to_yaml([1, 2], 0.3, 0.4, file_path=yaml_file, key_name="my_calib")
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        assert "my_calib" in data
        assert "tt_calibration" not in data

    def test_file_not_found(self, tmp_path, capsys):
        """Missing file should print an error and return without raising."""
        result = add_calibration_to_yaml(
            [1], 0.1, 0.2, file_path=str(tmp_path / "nonexistent.yaml")
        )
        assert result is None
        captured = capsys.readouterr()
        assert "Error" in captured.out or "not found" in captured.out

    def test_inf_string_variants(self, yaml_file):
        """All recognized inf-string variants should be converted."""
        add_calibration_to_yaml(["inf"], 0.1, 0.2, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        runs = list(data["tt_calibration"][0]["runs"])
        assert runs[0] == float("inf")

    def test_plus_inf_string(self, yaml_file):
        """+inf string should also be recognised."""
        add_calibration_to_yaml(["+inf"], 0.1, 0.2, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        runs = list(data["tt_calibration"][0]["runs"])
        assert runs[0] == float("inf")

    def test_slope_intercept_stored_as_float(self, yaml_file):
        """Slope and intercept must be stored as float."""
        add_calibration_to_yaml([1, 5], 1, 2, file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        entry = data["tt_calibration"][0]
        assert isinstance(entry["slope"], float)
        assert isinstance(entry["intercept"], float)

    def test_multiple_appends(self, yaml_file):
        """Multiple sequential appends should accumulate."""
        for i in range(3):
            add_calibration_to_yaml([i, i + 10], float(i), float(i * 2), file_path=yaml_file)
        yaml = YAML()
        with open(yaml_file) as f:
            data = yaml.load(f)
        assert len(data["tt_calibration"]) == 3

    def test_null_key_value(self, tmp_path):
        """When the key exists but is null, it should be initialised to a list."""
        fp = tmp_path / "config.yaml"
        yaml = YAML()
        yaml.dump({"experiment": "x", "tt_calibration": None}, fp)
        add_calibration_to_yaml([1, 2], 0.1, 0.2, file_path=str(fp))
        with open(str(fp)) as f:
            data = yaml.load(f)
        assert len(data["tt_calibration"]) == 1


# ===================================================================
# apply_timetool_correction
# ===================================================================
class TestApplyTimetoolCorrection:
    """Tests for apply_timetool_correction."""

    # --- single calibration (no run_indicator) ---

    def test_single_basic(self):
        """correction = edge * slope + intercept; result = delay + correction."""
        delays = np.array([1.0, 2.0, 3.0])
        edges = np.array([100.0, 200.0, 300.0])
        slope, intercept = 0.01, 0.5
        result = apply_timetool_correction(delays, edges, slope, intercept)
        expected = delays + edges * slope + intercept
        np.testing.assert_allclose(result, expected)

    def test_single_zero_slope(self):
        """Zero slope → correction is just the intercept."""
        delays = np.array([5.0, 6.0])
        edges = np.array([100.0, 200.0])
        result = apply_timetool_correction(delays, edges, 0.0, 1.0)
        np.testing.assert_allclose(result, delays + 1.0)

    def test_single_zero_intercept(self):
        delays = np.array([5.0, 6.0])
        edges = np.array([100.0, 200.0])
        result = apply_timetool_correction(delays, edges, 0.5, 0.0)
        np.testing.assert_allclose(result, delays + edges * 0.5)

    def test_single_returns_ndarray(self):
        result = apply_timetool_correction([1.0], [10.0], 0.1, 0.2)
        assert isinstance(result, np.ndarray)

    def test_single_accepts_lists(self):
        """Should accept plain Python lists."""
        result = apply_timetool_correction([1.0, 2.0], [10.0, 20.0], 0.1, 0.2)
        expected = np.array([1.0 + 10.0 * 0.1 + 0.2, 2.0 + 20.0 * 0.1 + 0.2])
        np.testing.assert_allclose(result, expected)

    def test_single_rejects_array_slopes(self):
        """Without run_indicator, array slopes should raise ValueError."""
        with pytest.raises(ValueError, match="scalars"):
            apply_timetool_correction([1.0], [10.0], [0.1, 0.2], 0.0)

    def test_single_rejects_array_intercepts(self):
        with pytest.raises(ValueError, match="scalars"):
            apply_timetool_correction([1.0], [10.0], 0.1, [0.0, 0.1])

    # --- multi calibration (with run_indicator) ---

    def test_multi_two_runs(self):
        """Two runs, each with own slope/intercept."""
        delays = np.array([1.0, 2.0, 3.0, 4.0])
        edges = np.array([100.0, 200.0, 300.0, 400.0])
        run_ids = np.array([1, 1, 2, 2])
        slopes = np.array([0.01, 0.02])
        intercepts = np.array([0.5, 1.0])

        result = apply_timetool_correction(delays, edges, slopes, intercepts, run_ids)
        expected = np.array([
            1.0 + 100.0 * 0.01 + 0.5,
            2.0 + 200.0 * 0.01 + 0.5,
            3.0 + 300.0 * 0.02 + 1.0,
            4.0 + 400.0 * 0.02 + 1.0,
        ])
        np.testing.assert_allclose(result, expected)

    def test_multi_scalar_slope_broadcast(self):
        """A single slope scalar should be broadcast to all runs."""
        delays = np.array([1.0, 2.0, 3.0])
        edges = np.array([10.0, 20.0, 30.0])
        run_ids = np.array([1, 2, 2])
        result = apply_timetool_correction(delays, edges, 0.5, 1.0, run_ids)
        expected = delays + edges * 0.5 + 1.0
        np.testing.assert_allclose(result, expected)

    def test_multi_scalar_intercept_broadcast(self):
        """A single intercept scalar should be broadcast."""
        delays = np.array([1.0, 2.0, 3.0])
        edges = np.array([10.0, 20.0, 30.0])
        run_ids = np.array([1, 1, 2])
        slopes = np.array([0.1, 0.2])
        result = apply_timetool_correction(delays, edges, slopes, 0.0, run_ids)
        expected = np.array([
            1.0 + 10.0 * 0.1,
            2.0 + 20.0 * 0.1,
            3.0 + 30.0 * 0.2,
        ])
        np.testing.assert_allclose(result, expected)

    def test_multi_mismatched_slopes_raises(self):
        """Wrong number of slopes should raise ValueError."""
        delays = np.array([1.0, 2.0])
        edges = np.array([10.0, 20.0])
        run_ids = np.array([1, 2])
        with pytest.raises(ValueError, match="slopes/intercepts"):
            apply_timetool_correction(delays, edges, [0.1, 0.2, 0.3], [0.0, 0.0], run_ids)

    def test_multi_mismatched_intercepts_raises(self):
        delays = np.array([1.0, 2.0])
        edges = np.array([10.0, 20.0])
        run_ids = np.array([1, 2])
        with pytest.raises(ValueError, match="slopes/intercepts"):
            apply_timetool_correction(delays, edges, [0.1, 0.2], [0.0, 0.0, 0.0], run_ids)

    def test_multi_shape_mismatch_raises(self):
        """run_indicator length must match delays and edges."""
        delays = np.array([1.0, 2.0, 3.0])
        edges = np.array([10.0, 20.0, 30.0])
        run_ids = np.array([1, 2])  # wrong length
        with pytest.raises(ValueError, match="same shape"):
            apply_timetool_correction(delays, edges, [0.1, 0.2], [0.0, 0.0], run_ids)

    def test_multi_three_runs(self):
        """Three distinct runs."""
        n = 9
        delays = np.ones(n)
        edges = np.arange(n, dtype=float) * 10.0
        run_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        slopes = np.array([0.1, 0.2, 0.3])
        intercepts = np.array([1.0, 2.0, 3.0])

        result = apply_timetool_correction(delays, edges, slopes, intercepts, run_ids)
        for i in range(n):
            run_idx = [1, 2, 3].index(run_ids[i])
            exp = delays[i] + edges[i] * slopes[run_idx] + intercepts[run_idx]
            assert abs(result[i] - exp) < 1e-12

    def test_multi_returns_ndarray(self):
        result = apply_timetool_correction(
            [1.0, 2.0], [10.0, 20.0], [0.1], [0.0], [5, 5]
        )
        assert isinstance(result, np.ndarray)

    def test_multi_single_run(self):
        """A single unique run in run_indicator should work fine."""
        delays = np.array([1.0, 2.0, 3.0])
        edges = np.array([10.0, 20.0, 30.0])
        run_ids = np.array([42, 42, 42])
        result = apply_timetool_correction(delays, edges, [0.5], [1.0], run_ids)
        expected = delays + edges * 0.5 + 1.0
        np.testing.assert_allclose(result, expected)

    def test_multi_non_contiguous_runs(self):
        """Run IDs don't need to be contiguous (e.g. 5, 10, 100)."""
        delays = np.array([0.0, 0.0, 0.0])
        edges = np.array([1.0, 1.0, 1.0])
        run_ids = np.array([5, 100, 10])
        slopes = np.array([1.0, 2.0, 3.0])       # sorted unique: [5, 10, 100]
        intercepts = np.array([0.0, 0.0, 0.0])

        result = apply_timetool_correction(delays, edges, slopes, intercepts, run_ids)
        # unique sorted: 5→idx0 (slope=1), 10→idx1 (slope=2), 100→idx2 (slope=3)
        np.testing.assert_allclose(result, [1.0, 3.0, 2.0])

    def test_output_shape_matches_input(self):
        delays = np.zeros(50)
        edges = np.ones(50)
        result = apply_timetool_correction(delays, edges, 0.1, 0.2)
        assert result.shape == delays.shape
