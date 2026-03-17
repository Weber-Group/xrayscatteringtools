"""Tests for xrayscatteringtools.theory.patterns."""

import numpy as np
import pytest
import h5py
from types import SimpleNamespace

from xrayscatteringtools.theory import patterns
from xrayscatteringtools.theory.patterns import (
    __all__ as PATTERNS_ALL,
    _make_default_obj,
    _make_default_docstring,
)
from xrayscatteringtools.utils import invAngstroms2au, au2invAngstroms


# ── Module-level attribute access (lazy loading) ───────────────────────


class TestPatternLazyLoading:
    """Each name in __all__ should load a valid pattern object."""

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_loads_without_error(self, name):
        obj = getattr(patterns, name)
        assert obj is not None

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_has_required_attributes(self, name):
        obj = getattr(patterns, name)
        assert hasattr(obj, "q")
        assert hasattr(obj, "I_q")
        assert hasattr(obj, "molecule")
        assert hasattr(obj, "method")
        assert hasattr(obj, "basis_set")
        assert hasattr(obj, "n_electrons")

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_q_and_Iq_same_length(self, name):
        obj = getattr(patterns, name)
        assert len(obj.q) == len(obj.I_q)

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_q_positive_and_sorted(self, name):
        obj = getattr(patterns, name)
        assert np.all(obj.q >= 0)
        assert np.all(np.diff(obj.q) > 0)

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_Iq_positive(self, name):
        obj = getattr(patterns, name)
        assert np.all(obj.I_q >= 0)

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_elastic_inelastic_lengths_if_present(self, name):
        obj = getattr(patterns, name)
        if hasattr(obj, "I_q_elastic"):
            assert len(obj.I_q_elastic) == len(obj.q)
        if hasattr(obj, "I_q_inelastic"):
            assert len(obj.I_q_inelastic) == len(obj.q)

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_has_docstring(self, name):
        obj = getattr(patterns, name)
        assert obj.__doc__ is not None
        assert len(obj.__doc__) > 50

    def test_invalid_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(patterns, "NONEXISTENT_PATTERN_XYZ")


# ── __dir__ ────────────────────────────────────────────────────────────


class TestDir:
    def test_dir_returns_all_sorted(self):
        result = patterns.__dir__()
        assert result == sorted(PATTERNS_ALL)

    def test_dir_contains_expected(self):
        result = patterns.__dir__()
        assert "SF6__CCSD__aug_cc_pVDZ" in result


# ── _make_default_obj ──────────────────────────────────────────────────


class TestMakeDefaultObj:
    @pytest.fixture
    def h5_with_elastic(self, tmp_path):
        """Create a temporary HDF5 pattern file with elastic/inelastic data."""
        path = str(tmp_path / "test_pattern.h5")
        q = np.linspace(0.1, 10.0, 50)
        with h5py.File(path, "w") as f:
            f.create_dataset("q", data=q)
            f.create_dataset("I_q", data=np.ones_like(q) * 10)
            f.create_dataset("I_q_elastic", data=np.ones_like(q) * 7)
            f.create_dataset("I_q_inelastic", data=np.ones_like(q) * 3)
            f.attrs["molecule"] = "Test"
            f.attrs["method"] = "HF"
            f.attrs["basis_set"] = "STO-3G"
            f.attrs["n_electrons"] = 10
        return path

    @pytest.fixture
    def h5_without_elastic(self, tmp_path):
        """Create a temporary HDF5 pattern file without elastic/inelastic."""
        path = str(tmp_path / "test_pattern_minimal.h5")
        q = np.linspace(0.1, 10.0, 50)
        with h5py.File(path, "w") as f:
            f.create_dataset("q", data=q)
            f.create_dataset("I_q", data=np.ones_like(q) * 10)
            f.attrs["molecule"] = "Minimal"
            f.attrs["method"] = "DFT"
            f.attrs["basis_set"] = "cc-pVDZ"
            f.attrs["n_electrons"] = 8
        return path

    def test_loads_all_fields(self, h5_with_elastic):
        with h5py.File(h5_with_elastic, "r") as f:
            obj = _make_default_obj(f)
        assert len(obj.q) == 50
        assert obj.molecule == "Test"
        assert obj.method == "HF"
        assert obj.basis_set == "STO-3G"
        assert obj.n_electrons == 10
        assert hasattr(obj, "I_q_elastic")
        assert hasattr(obj, "I_q_inelastic")
        np.testing.assert_allclose(obj.I_q_elastic, 7)
        np.testing.assert_allclose(obj.I_q_inelastic, 3)

    def test_optional_elastic_inelastic(self, h5_without_elastic):
        with h5py.File(h5_without_elastic, "r") as f:
            obj = _make_default_obj(f)
        assert hasattr(obj, "q")
        assert hasattr(obj, "I_q")
        assert not hasattr(obj, "I_q_elastic")
        assert not hasattr(obj, "I_q_inelastic")


# ── _make_default_docstring ────────────────────────────────────────────


class TestMakeDefaultDocstring:
    def test_docstring_with_elastic(self):
        obj = SimpleNamespace(
            q=np.linspace(0.1, 10.0, 50),
            I_q=np.ones(50),
            I_q_elastic=np.ones(50),
            I_q_inelastic=np.ones(50),
            molecule="SF6",
            method="CCSD",
            basis_set="aug-cc-pVDZ",
            n_electrons=70,
        )
        doc = _make_default_docstring(obj)
        assert "SF6" in doc
        assert "CCSD" in doc
        assert "aug-cc-pVDZ" in doc
        assert "I_q_elastic" in doc
        assert "I_q_inelastic" in doc
        assert "n_electrons" in doc

    def test_docstring_without_elastic(self):
        obj = SimpleNamespace(
            q=np.linspace(0.1, 10.0, 50),
            I_q=np.ones(50),
            molecule="H2O",
            method="HF",
            basis_set="STO-3G",
            n_electrons=10,
        )
        doc = _make_default_docstring(obj)
        assert "H2O" in doc
        assert "HF" in doc
        assert "I_q_elastic" not in doc
        assert "I_q_inelastic" not in doc
        assert "n_electrons" in doc

    def test_docstring_contains_unit_conversion(self):
        obj = SimpleNamespace(
            q=np.linspace(0.5, 8.0, 30),
            I_q=np.ones(30),
            molecule="Ne",
            method="MP2",
            basis_set="cc-pVDZ",
            n_electrons=10,
        )
        doc = _make_default_docstring(obj)
        # Should contain the a.u. conversion factor
        assert str(au2invAngstroms(1.0)) in doc


# ── Consistency between shipped patterns ───────────────────────────────


class TestShippedPatternConsistency:
    """Cross-checks on the bundled ab-initio data."""

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_elastic_plus_inelastic_equals_total(self, name):
        obj = getattr(patterns, name)
        if not (hasattr(obj, "I_q_elastic") and hasattr(obj, "I_q_inelastic")):
            pytest.skip(f"{name} does not have separate elastic/inelastic data")
        np.testing.assert_allclose(
            obj.I_q, obj.I_q_elastic + obj.I_q_inelastic,
            rtol=1e-10,
            err_msg=f"elastic + inelastic != total for {name}",
        )

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_n_electrons_positive(self, name):
        obj = getattr(patterns, name)
        assert obj.n_electrons > 0

    @pytest.mark.parametrize("name", PATTERNS_ALL)
    def test_molecule_non_empty(self, name):
        obj = getattr(patterns, name)
        assert isinstance(obj.molecule, str)
        assert len(obj.molecule) > 0
